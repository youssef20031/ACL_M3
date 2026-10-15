import { Link } from 'react-router-dom';
import { MessageCircle } from 'lucide-react';

/** Sticky call to action on phones and tablets: the assistant is one tap away from any page. */
export function MobileAskButton() {
  return (
    <Link
      to="/assistant"
      aria-label="Ask the assistant"
      className="fixed bottom-[calc(1rem+env(safe-area-inset-bottom))] right-4 z-40 inline-flex h-12 items-center gap-2 rounded-full bg-accent px-5 font-semibold text-accent-on shadow-pop transition-transform hover:scale-[1.03] focus-visible:outline-offset-4 lg:hidden"
    >
      <MessageCircle className="h-5 w-5" aria-hidden="true" />
      Ask AI
    </Link>
  );
}
