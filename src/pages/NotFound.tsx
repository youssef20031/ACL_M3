import { Link } from 'react-router-dom';
import { Button, Container, EmptyState } from '../components/ui';

export function NotFound() {
  return (
    <Container>
      <EmptyState
        className="py-24"
        title="Page not found"
        description="That page doesn't exist, or it has moved."
        action={
          <Link to="/">
            <Button tabIndex={-1}>Back to home</Button>
          </Link>
        }
      />
    </Container>
  );
}
