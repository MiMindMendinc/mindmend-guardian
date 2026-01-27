# Contributing to MindMend Guardian

Thanks for looking — I'm a solo Michigan dad building MindMend Guardian, a privacy-first, offline child-safety AI. Family, farm, and nonprofit life mean I'm short on time, so any help is deeply appreciated.

How you can help
- Small docs fixes (README, install steps)
- Tests or CI improvements
- Raspberry Pi / low-power install recipes
- Voice phrases, sample alerts, and UX wording
- Code review, debugging, or feature PRs

Getting started
1. Fork the repo  
2. Clone and set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   # Install dependencies when available (e.g., pip install -r requirements.txt)
   git submodule update --init --recursive
   ```
3. Create a feature branch: `git checkout -b feature/short-description`  
4. Make changes and add tests where applicable  
5. Commit with clear messages: `git commit -m "Short: what changed"`  
6. Push: `git push origin feature/short-description`  
7. Open a Pull Request against `main`

Testing
- Run tests with: `pytest -q`
- Mark device/hardware-specific tests with `@pytest.mark.hardware`
- Run hardware tests only on Raspberry Pi or when hardware is available
- Ensure all tests pass before submitting a PR

Communication
- Open issues for questions or small tasks
- Tag me on X: @p_perrien if you want a quick ping

Perks & thanks
- First contributors (meaningful PRs merged) get a named farm animal (cow, goat, or chicken) and a shoutout in the README. 🐄🐐🐔
- Contributors who help significantly will be added to the project acknowledgements.

Code of conduct & safety
- This project is about protecting kids; please be respectful and collaborative.
- We follow our [Code of Conduct](CODE_OF_CONDUCT.md). For reference, we align with principles similar to the Contributor Covenant v2.1 — https://www.contributor-covenant.org/version/2/1/code_of_conduct/

Thank you — your time matters here.