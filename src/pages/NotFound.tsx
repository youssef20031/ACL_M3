import { ButtonLink, Container, EmptyState } from '../components/ui';

export function NotFound() {
  return (
    <Container>
      <EmptyState
        className="py-24"
        title="Page not found"
        description="That page doesn't exist, or it has moved."
        action={<ButtonLink to="/">Back to home</ButtonLink>}
      />
    </Container>
  );
}
