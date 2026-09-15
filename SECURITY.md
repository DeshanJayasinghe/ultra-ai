# Security Policy

WearWise treats AI service security and privacy as launch-critical.

## Reporting

Report suspected vulnerabilities privately to the project maintainers. Do not open public issues for vulnerabilities.

## Requirements

* Never commit secrets.
* Treat uploaded media as untrusted.
* Validate image inputs before processing.
* Authenticate internal service requests.
* Do not log personal data, signed URLs, image payloads, or provider secrets.
* Keep model outputs auditable through model versions and confidence metadata.
