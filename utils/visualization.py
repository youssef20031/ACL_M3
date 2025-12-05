"""
Knowledge Graph Visualization Component
Creates interactive graph visualizations using NetworkX and Plotly
"""
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import random


class GraphVisualizer:
    """Creates interactive visualizations of the FPL Knowledge Graph."""
    
    # Color schemes for different node types
    NODE_COLORS = {
        "Player": "#4CAF50",  # Green
        "Team": "#2196F3",    # Blue
        "Position": "#FF9800", # Orange
        "Season": "#9C27B0",  # Purple
        "Gameweek": "#607D8B", # Gray
        "Fixture": "#F44336"  # Red
    }
    
    def __init__(self, connection=None):
        """
        Initialize visualizer.
        
        Args:
            connection: Neo4j connection instance
        """
        self.conn = connection
    
    def build_subgraph_from_query(
        self, 
        nodes: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> nx.Graph:
        """
        Build NetworkX graph from query results.
        
        Args:
            nodes: List of node dictionaries
            relationships: List of relationship dictionaries
            
        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            node_id = node.get("id") or node.get("name") or str(id(node))
            node_type = node.get("type", "Unknown")
            G.add_node(
                node_id,
                label=node.get("name", node_id),
                node_type=node_type,
                **node
            )
        
        # Add edges
        for rel in relationships:
            source = rel.get("source") or rel.get("from")
            target = rel.get("target") or rel.get("to")
            rel_type = rel.get("type", "RELATED")
            
            if source and target:
                G.add_edge(source, target, relationship=rel_type, **rel)
        
        return G
    
    def fetch_player_graph(self, player_name: str, season: str = "2022-23") -> nx.Graph:
        """
        Fetch subgraph for a specific player.
        
        Args:
            player_name: Player name
            season: Season to query
            
        Returns:
            NetworkX Graph
        """
        if not self.conn:
            return nx.Graph()
        
        query = """
        MATCH (p:Player {name: $player_name})-[:PLAYS_POSITION]->(pos:Position)
        MATCH (p)-[played:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        MATCH (f)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        RETURN p, pos, f, gw, s, ht, at, played
        LIMIT 20
        """
        
        G = nx.Graph()
        
        try:
            results = self.conn.execute_query(query, {"player_name": player_name, "season": season})
            
            # Build graph from results
            G.add_node(player_name, node_type="Player", label=player_name)
            
            for record in results:
                # This is simplified - actual implementation would parse Neo4j records
                pass
                
        except Exception as e:
            print(f"Error fetching player graph: {e}")
        
        return G
    
    def create_plotly_figure(
        self, 
        G: nx.Graph,
        title: str = "Knowledge Graph Visualization",
        layout: str = "spring"
    ) -> go.Figure:
        """
        Create interactive Plotly figure from NetworkX graph.
        
        Args:
            G: NetworkX Graph
            title: Figure title
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            
        Returns:
            Plotly Figure object
        """
        if len(G.nodes()) == 0:
            return self._empty_figure(title)
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces (one per type for legend)
        node_traces = []
        
        # Group nodes by type
        nodes_by_type = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('node_type', 'Unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        for node_type, nodes in nodes_by_type.items():
            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            node_text = [G.nodes[node].get('label', node) for node in nodes]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                name=node_type,
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=20,
                    color=self.NODE_COLORS.get(node_type, '#999'),
                    line_width=2
                )
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        return fig
    
    def _empty_figure(self, title: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def create_schema_visualization(self) -> go.Figure:
        """
        Create visualization of the database schema.
        
        Returns:
            Plotly Figure showing schema
        """
        G = nx.DiGraph()
        
        # Add node types
        nodes = ["Player", "Team", "Position", "Season", "Gameweek", "Fixture"]
        for node in nodes:
            G.add_node(node, node_type=node, label=node)
        
        # Add relationships
        relationships = [
            ("Player", "Position", "PLAYS_POSITION"),
            ("Player", "Team", "PLAYS_FOR"),
            ("Player", "Fixture", "PLAYED_IN"),
            ("Fixture", "Team", "HOME_TEAM"),
            ("Fixture", "Team", "AWAY_TEAM"),
            ("Fixture", "Gameweek", "PART_OF"),
            ("Gameweek", "Season", "IN_SEASON")
        ]
        
        for source, target, rel_type in relationships:
            G.add_edge(source, target, relationship=rel_type)
        
        return self.create_plotly_figure(G, title="FPL Knowledge Graph Schema", layout="circular")
    
    def create_stats_chart(
        self,
        data: List[Dict[str, Any]],
        x_field: str,
        y_field: str,
        chart_type: str = "bar",
        title: str = "Statistics"
    ) -> go.Figure:
        """
        Create statistical chart from query results.
        
        Args:
            data: List of data dictionaries
            x_field: Field name for x-axis
            y_field: Field name for y-axis
            chart_type: Type of chart ('bar', 'line', 'scatter')
            title: Chart title
            
        Returns:
            Plotly Figure
        """
        if not data:
            return self._empty_figure(title)
        
        x_values = [d.get(x_field, "") for d in data]
        y_values = [d.get(y_field, 0) for d in data]
        
        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(x=x_values, y=y_values, marker_color='#4CAF50')
            ])
        elif chart_type == "line":
            fig = go.Figure(data=[
                go.Scatter(x=x_values, y=y_values, mode='lines+markers')
            ])
        elif chart_type == "scatter":
            fig = go.Figure(data=[
                go.Scatter(x=x_values, y=y_values, mode='markers')
            ])
        else:
            fig = go.Figure(data=[
                go.Bar(x=x_values, y=y_values)
            ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_field.replace("_", " ").title(),
            yaxis_title=y_field.replace("_", " ").title()
        )
        
        return fig
    
    def create_player_radar(
        self,
        player_data: Dict[str, Any],
        metrics: List[str] = None
    ) -> go.Figure:
        """
        Create radar chart for player statistics.
        
        Args:
            player_data: Player statistics dictionary
            metrics: List of metrics to include
            
        Returns:
            Plotly Figure with radar chart
        """
        if metrics is None:
            metrics = ["goals", "assists", "bonus", "clean_sheets", "minutes"]
        
        values = [player_data.get(m, 0) for m in metrics]
        values.append(values[0])  # Close the polygon
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=player_data.get("name", "Player")
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
            showlegend=True,
            title=f"{player_data.get('name', 'Player')} Statistics"
        )
        
        return fig
