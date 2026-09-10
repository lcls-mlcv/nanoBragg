/*
 * argkey.h -- canonicalize a command-line argument list for stable hashing.
 *
 * Reorders an argument list into a canonical form so a cache key derived from
 * it is independent of the order the flags were written in, with no per-flag
 * arity table (NBTOOLS-SPEC §12). Applied only to the hash input; the stored
 * and executed args are untouched.
 *
 * Classification: a token is a FLAG iff it starts with '-' (or leading dashes)
 * followed by a letter ("-cell", "--misset"); everything else is a VALUE
 * ("74", "/path", "-10", "-.5", lone "-"). A BUNDLE is a flag plus every
 * following non-flag token up to the next flag; a leading non-flag run forms
 * one bundle keyed on "" that sorts first. Bundles stable-sort by flag name;
 * values within a bundle keep their original order. No token is ever dropped
 * or merged, so distinct arg sets can never collide (worst case a spurious
 * re-render, never a wrong cache hit).
 *
 * Caveat: the '-'+letter rule treats a digit-leading flag (nanoBragg's
 * "-4stol") as a value. Benign for keying -- still deterministic and
 * collision-free.
 */
#ifndef ARGKEY_H
#define ARGKEY_H

/*
 * Canonicalize tok[0..n) and return the result as a newly heap-allocated,
 * NUL-terminated string: the canonical tokens space-joined, with NO trailing
 * newline. Empty input (n <= 0) returns an empty string "". The input token
 * pointers and their contents are not modified.
 *
 * OWNERSHIP: the caller owns the returned buffer and must free() it. Returns
 * NULL only on allocation failure.
 */
char *argkey_canonicalize(char **tok, int n);

#endif /* ARGKEY_H */
