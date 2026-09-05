# Manual release decisions

## Software license — MANUAL DECISION REQUIRED

No `LICENSE`, `LICENSE.txt`, or `COPYING` file exists. The repository owner must
choose whether and how the original source code is licensed. No license was selected
or inferred by this task.

A software license applies only to material the owner has authority to license. It
does not itself grant redistribution rights for third-party `.osr` replay files,
`.osu` beatmaps, API responses, usernames, or other source data.

## Third-party data and privacy — MANUAL RIGHTS REVIEW

Before any raw data or row-level result table is published, a human must establish
its origin, applicable terms, redistribution permission, attribution requirements,
and an appropriate treatment of player-linked identifiers. This inventory is
engineering guidance, not legal advice.

## Artifact distribution

The minimal GitHub tree excludes the large branch inputs needed for full v2 audit
regeneration. Decide whether to publish a separate hash-verified artifact archive or
release asset after privacy/rights review. Git LFS should be considered only after
that decision; it does not resolve licensing or privacy.

## Supported environment

Choose and document the supported Python/platform matrix and dependency pinning
policy. The current requirements declare direct packages but are not a locked
environment. Figure generation currently calls macOS `sips`, so a portable public
figure-generation strategy or a documented platform limitation is still required.

## README and citation

Approve the final public claim wording, author/citation metadata, data-availability
statement, and license status before writing the final README.

