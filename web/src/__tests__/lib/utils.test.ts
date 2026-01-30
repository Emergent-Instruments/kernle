import { describe, it, expect } from 'vitest';
import { cn } from '../../lib/utils';

describe('cn utility', () => {
  it('merges class names', () => {
    expect(cn('foo', 'bar')).toBe('foo bar');
  });

  it('handles conditional classes with clsx', () => {
    expect(cn('base', true && 'included', false && 'excluded')).toBe('base included');
  });

  it('merges tailwind classes correctly', () => {
    // twMerge should keep last conflicting class
    expect(cn('px-2', 'px-4')).toBe('px-4');
    expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500');
  });

  it('handles arrays of classes', () => {
    expect(cn(['foo', 'bar'])).toBe('foo bar');
  });

  it('handles undefined and null', () => {
    expect(cn('base', undefined, null, 'other')).toBe('base other');
  });

  it('handles empty inputs', () => {
    expect(cn()).toBe('');
    expect(cn('')).toBe('');
  });

  it('handles object syntax', () => {
    expect(cn({ active: true, disabled: false })).toBe('active');
  });

  it('handles complex tailwind merge scenarios', () => {
    // Margin overrides
    expect(cn('m-2', 'mx-4')).toBe('m-2 mx-4');
    expect(cn('mx-2', 'mx-4')).toBe('mx-4');

    // Responsive variants
    expect(cn('text-sm', 'md:text-lg')).toBe('text-sm md:text-lg');

    // State variants
    expect(cn('hover:bg-blue-500', 'hover:bg-red-500')).toBe('hover:bg-red-500');
  });
});
