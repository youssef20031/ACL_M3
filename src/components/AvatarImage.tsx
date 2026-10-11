import { useEffect, useState, type ReactNode } from 'react';

interface AvatarImageProps {
  src?: string | null;
  alt: string;
  className?: string;
  /** Intrinsic size hint (px) so the browser can reserve space before the image loads. */
  size?: number;
  /** Rendered when there is no src or the image fails to load. */
  fallback?: ReactNode;
  /** Defaults to lazy; pass 'eager' for above-the-fold hero images. */
  loading?: 'lazy' | 'eager';
}

/**
 * Plain <img> with lazy/async decoding and an onError fallback.
 * Purely behavioural: callers keep passing their existing classes and fallback markup.
 */
export function AvatarImage({ src, alt, className, size, fallback = null, loading = 'lazy' }: AvatarImageProps) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  if (!src || failed) {
    return <>{fallback}</>;
  }

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      width={size}
      height={size}
      loading={loading}
      decoding="async"
      onError={() => setFailed(true)}
    />
  );
}
