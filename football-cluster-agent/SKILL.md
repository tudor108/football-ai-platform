# Dockerfile Best Practices Skill

## Purpose
A quick, reusable checklist for reviewing and creating Dockerfiles according to best practices. Intended for use across all projects.

## Checklist
- [ ] Use official base images and specify exact tags (avoid `latest`)
- [ ] Minimize the number of layers (combine commands with `&&` where possible)
- [ ] Use `.dockerignore` to exclude unnecessary files
- [ ] Leverage multi-stage builds to reduce image size
- [ ] Avoid installing unnecessary packages
- [ ] Set `WORKDIR` instead of using `cd`
- [ ] Use `COPY` and `ADD` efficiently (prefer `COPY` for files, `ADD` only for URLs/tar)
- [ ] Set non-root user with `USER` where possible
- [ ] Use `HEALTHCHECK` for critical services
- [ ] Clean up caches and temp files in the same layer as install
- [ ] Set environment variables with `ENV` for configuration
- [ ] Document exposed ports with `EXPOSE`
- [ ] Specify entrypoint and/or command with `ENTRYPOINT` and `CMD`
- [ ] Avoid secrets in the image or build context
- [ ] Pin dependency versions in install commands

## Example Prompts
- "Review this Dockerfile for best practices."
- "Generate a Dockerfile following best practices."
- "Checklist for optimizing Docker images."

## Related Customizations
- Multi-stage build workflow skill
- Security-focused Dockerfile review skill
- Automated Docker image size analysis skill
