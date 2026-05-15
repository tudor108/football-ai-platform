# Dockerfile Security Check Skill

## Purpose
A quick, reusable checklist for performing security reviews on Dockerfiles. Intended for use across all projects.

## Checklist
- [ ] Use minimal base images (e.g., `alpine`, `distroless`)
- [ ] Avoid running as root; set a non-root `USER`
- [ ] Do not include secrets, credentials, or sensitive files in the image or build context
- [ ] Use `.dockerignore` to exclude sensitive files (e.g., `.env`, SSH keys)
- [ ] Pin versions for OS and package dependencies
- [ ] Regularly update base images and dependencies to patch vulnerabilities
- [ ] Remove build tools and package managers in final image (multi-stage builds)
- [ ] Set appropriate file and directory permissions
- [ ] Use `HEALTHCHECK` to monitor container health
- [ ] Limit container capabilities (e.g., with `--cap-drop` at runtime)
- [ ] Avoid `ADD` for remote URLs (prefer `COPY`)
- [ ] Validate and sanitize all inputs to the container
- [ ] Expose only necessary ports
- [ ] Use trusted sources for all downloads and dependencies
- [ ] Scan images for vulnerabilities before deployment (e.g., Trivy, Snyk)

## Example Prompts
- "Review this Dockerfile for security risks."
- "Checklist for Dockerfile security best practices."
- "How can I secure my Docker image?"

## Related Customizations
- Automated vulnerability scanning skill
- Secrets detection skill
- Secure base image selection skill
