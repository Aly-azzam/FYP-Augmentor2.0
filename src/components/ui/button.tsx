import * as React from 'react';
import { cn } from '../../utils/cn';

const variantClasses = {
  default: 'btn btn-primary',
  secondary: 'btn btn-secondary',
  ghost: 'btn btn-ghost',
  outline:
    'btn bg-transparent border border-[var(--border-default)] text-[var(--text-primary)] hover:border-[var(--border-hover)] hover:bg-[var(--accent-soft)]',
  destructive: 'btn bg-[var(--error)] text-white hover:bg-red-600',
} as const;

const sizeClasses = {
  sm: 'px-3 py-1.5 text-xs',
  default: '',
  lg: 'px-6 py-3 text-base',
  icon: 'p-2 aspect-square',
} as const;

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variantClasses;
  size?: keyof typeof sizeClasses;
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    return (
      <button
        className={cn(variantClasses[variant], sizeClasses[size], className)}
        ref={ref}
        {...props}
      />
    );
  },
);

Button.displayName = 'Button';

export { Button };
