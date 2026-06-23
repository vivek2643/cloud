---
name: frontend-design
description: >-
  Design-system and conventions for this app's frontend: a minimalist
  black/white/grey aesthetic with orange used sparingly as the only accent.
  Use whenever creating or editing anything under frontend/ — React components,
  pages, styling, layout, Tailwind classes, or theme tokens.
---

# Frontend Design System

Minimalist, black-themed product UI. Restraint over decoration.

## Aesthetic principles

- **Palette is black / white / grey.** Orange (`--accent`) is the ONLY color and
  appears sparingly — primary buttons, the occasional active indicator, a single
  highlight. If a screen has more than ~2 orange elements, remove some.
- **Minimalism first.** Prefer empty space to dividers, weight/spacing to color,
  one clear primary action per view. No gradients, no shadows-as-decoration, no
  multi-color UI.
- **Quiet motion.** Use `transition-colors` / short `duration-150..200`. No
  bouncy or attention-seeking animation.
- **Hierarchy via type + space**, not boxes. Use `--muted` for secondary text,
  `--foreground` for primary, weight (`font-medium`) for emphasis.

## Color tokens (single source of truth)

All colors live as CSS variables in `frontend/src/app/globals.css`. **Never
hardcode hex values in components.** Reference tokens via inline style, e.g.
`style={{ color: "var(--muted)", background: "var(--sidebar)" }}`.

| Token | Role |
|-------|------|
| `--background` | Page background (near-black) |
| `--foreground` | Primary text/icons (near-white) |
| `--sidebar` | Raised surfaces: sidebars, cards, panels |
| `--border` | Hairline borders + subtle hover fills |
| `--muted` | Secondary/disabled text (grey) |
| `--accent` | Orange — primary buttons & rare highlights only |
| `--accent-hover` | Orange hover state |
| `--accent-soft` | Faint orange wash (selected/active backgrounds) |
| `--danger` | Destructive actions (burnt orange, stays in palette) |
| `--success` | Positive state |

The brand orange (from the EDSO logo) is **`#ed5b00`** — set on `--accent` in
`globals.css`. Change it in that ONE place and never inline it elsewhere.

## Stack facts (don't re-derive these)

- **Next.js 15 App Router + React 19.** Client components need `"use client";`.
- **Tailwind CSS v4, CSS-first.** Config lives in `globals.css` (`@import
  "tailwindcss"`). There is **no `tailwind.config.js`** — add theme tokens as CSS
  variables in `globals.css`, not a JS config.
- **Class merging:** `cn()` from `@/lib/utils` (clsx + tailwind-merge). Use it for
  any conditional className.
- **Icons:** `lucide-react`, default `size={16}` for inline UI icons.
- **State:** `zustand`.
- **Components:** `frontend/src/components/`, kebab-case filenames
  (`hero-cuts-view.tsx`), exported as named PascalCase functions.

## Component conventions

- Tailwind utilities for **layout/spacing/typography**; CSS-var inline styles for
  **color**. (Layout = classes, color = tokens.)
- Default radius `rounded-lg`; default body text `text-sm`.
- Borders are hairline: `border` + `style={{ borderColor: "var(--border)" }}`.
- Buttons: primary = `--accent` bg + `--background` text; secondary = transparent
  with `--border`; ghost = transparent, `--muted` text → `--foreground` on hover.
- Hover/active fills use `--border` or `--accent-soft`, never a new color.

## Checklist before finishing UI work

- [ ] No hardcoded hex — all colors via `var(--token)`
- [ ] Orange used in ≤ a couple of places per view
- [ ] `cn()` used for conditional classes
- [ ] No `tailwind.config.js` introduced; tokens stay in `globals.css`
- [ ] Spacing/whitespace doing the work instead of borders/shadows
