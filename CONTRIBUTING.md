
# Contributing to the Project

Thank you for your interest in contributing.

The goal of this project is to develop a **modern, modular, extensible, and research-oriented DDPM implementation in PyTorch**.

This project is not intended to be a collection of disconnected implementations of diffusion algorithms. Every feature should contribute to a unified diffusion framework where different architectures, training strategies, noise schedules, sampling algorithms, and optimization techniques work together through a consistent pipeline.

Contributions should prioritize correctness, modularity, seamless integration, reproducibility, and maintainability.

Please read these guidelines before submitting a contribution.

---

## 1. Project Philosophy

The primary objective is to evolve the existing implementation into a modern DDPM framework that incorporates improvements from contemporary diffusion research.

Every contribution should follow these principles:

- **Seamless Integration:** New features must integrate directly into the existing training, validation, checkpointing, and sampling pipelines.
- **Modularity:** Implementations should be modular and reusable rather than tightly coupled to a specific model, dataset, or workflow.
- **Configurability:** Features must be configurable through the existing configuration system and exposed through appropriate command-line arguments.
- **Backward Compatibility:** Existing functionality and workflows should continue working unless a breaking change is explicitly discussed and approved.
- **Research Correctness:** Implementations must follow the mathematical formulations and intended behavior described in their respective research papers.
- **Reproducibility:** Training and sampling behavior should be reproducible through documented configurations and checkpoints.
- **Testing:** Every contribution must be tested before a pull request is submitted.
- **Maintainability:** Code should follow the existing project structure and avoid unnecessary complexity or duplication.

A feature is not considered complete merely because its core algorithm works independently. It must also be integrated, configurable, tested, and documented.

---

## 2. Seamless Pipeline Integration

This is one of the most important requirements of the project.

New implementations must extend the existing framework rather than introduce independent scripts or duplicate existing functionality.

For example:

- New noise schedules should integrate with the existing noise scheduler.
- New loss functions should integrate with the existing training and validation workflows.
- New sampling algorithms should integrate with the existing sampling pipeline.
- New model architectures should work with the existing training and checkpointing mechanisms.
- New conditioning mechanisms should integrate with the existing dataset, model, and sampling components.
- New optimization techniques should integrate with the existing Trainer.

### Requirements

- Reuse existing components wherever possible.
- Avoid creating separate training pipelines for individual features.
- Avoid creating separate sampling scripts when the functionality belongs in the existing sampling pipeline.
- Avoid duplicating functionality that is already implemented.
- Preserve existing component interfaces wherever practical.
- Ensure new components can be enabled or disabled without affecting unrelated functionality.
- Ensure different features can work together when their functionality is compatible.

**The expected result is a unified framework, not multiple independent implementations sharing the same repository.**

---

## 3. Configuration and Command-Line Integration

Every new feature must be configurable.

The project already provides a configuration system and command-line interfaces. Contributions must extend these existing mechanisms rather than introduce hardcoded feature selections.

### Configuration Requirements

- Add appropriate configuration options for every new feature.
- Provide sensible default values.
- Preserve existing defaults unless a change is explicitly justified.
- Validate configuration values.
- Handle unsupported or incompatible configurations gracefully.
- Keep configuration names consistent with the existing project conventions.

### Command-Line Requirements

New functionality must be accessible through the appropriate existing command-line interface.

For example:

- Training-related features should be configurable through `train.py`.
- Sampling-related features should be configurable through `sample.py`.
- Evaluation-related features should be exposed through the relevant evaluation interface.

The command-line interface should:

- Expose relevant feature settings through configurable arguments.
- Allow command-line arguments to override their corresponding configuration values.
- Preserve configuration defaults when arguments are omitted.
- Validate user-provided arguments.
- Provide clear error messages for invalid configurations.
- Avoid introducing unnecessary command-line arguments for internal implementation details.

**Important:** A feature that can only be activated by modifying source code or manually editing internal implementation files is not considered fully integrated.

---

## 4. Code Quality and Architecture

Contributions should follow the existing architecture and coding conventions.

### Requirements

- Use PyTorch and the existing project dependencies wherever possible.
- Keep individual components focused on clearly defined responsibilities.
- Prefer reusable modules over duplicated implementations.
- Avoid unnecessary dependencies.
- Use meaningful variable, class, and function names.
- Include appropriate documentation and type annotations where useful.
- Handle device placement correctly.
- Ensure compatibility with the existing mixed-precision training workflow where applicable.
- Avoid unnecessary modifications to unrelated components.
- Handle edge cases and invalid inputs explicitly.

### Mathematical Implementations

For contributions based on research papers:

- Follow the mathematical formulation described in the reference paper.
- Clearly distinguish between alternative formulations and implementation-specific optimizations.
- Document important mathematical assumptions.
- Handle numerical stability appropriately.
- Avoid silently changing the intended objective or algorithm.
- Include references to the original research paper.

If an implementation differs from the reference paper, explain the deviation and its justification.

---

## 5. Compatibility and Backward Compatibility

New features should not unintentionally break existing functionality.

Where applicable, contributions must preserve compatibility with:

- Existing DDPM training.
- Existing sampling algorithms.
- Pixel-space diffusion.
- Latent-space diffusion.
- Existing noise schedules.
- Existing loss functions.
- EMA.
- Mixed-precision training.
- Gradient clipping.
- Existing checkpoint formats.
- Existing model configurations.
- Conditional and unconditional generation.

Not every feature is necessarily compatible with every other feature. When an incompatibility exists, it must be explicitly handled, documented, and validated rather than silently ignored.

### Checkpoint Compatibility

- Preserve existing checkpoint loading behavior wherever possible.
- Save any additional state required by the new feature.
- Restore the relevant configuration and model state during inference.
- Do not assume that checkpoints trained with different architectures or objectives are interchangeable.
- Provide appropriate handling for older checkpoints when backward compatibility is intended.

---

## 6. Testing Requirements

**Testing is mandatory before submitting a pull request.**

A contribution should not be submitted solely because the implementation runs without raising an exception.

Contributors are expected to verify both the new functionality and its integration with the existing framework.

### General Testing Requirements

- Test the new feature independently.
- Test its integration with the existing pipeline.
- Verify that configuration options work correctly.
- Verify command-line argument overrides.
- Verify default behavior when optional arguments are omitted.
- Test invalid configurations and edge cases.
- Check for NaN and infinite values where numerical computations are involved.
- Verify that existing functionality remains unaffected.
- Include appropriate unit tests and integration tests.

### Training-Related Changes

For training-related contributions, verify:

- Forward and backward passes.
- Gradient propagation.
- Loss calculation.
- Optimizer integration.
- Validation behavior.
- Checkpoint saving and restoration.
- EMA compatibility where applicable.
- Mixed-precision compatibility where applicable.

### Sampling-Related Changes

For sampling-related contributions, verify:

- Correct inference behavior.
- Correct timestep handling.
- Numerical stability.
- Configurable sampling settings.
- EMA and original weight selection.
- Pixel-space and latent-space compatibility where applicable.
- Integration with existing sampling algorithms.
- Generated sample quality and output correctness.

### Performance-Related Changes

When a contribution claims performance improvements, include meaningful measurements.

Where applicable, report:

- Hardware used.
- Relevant configuration settings.
- Inference or training time.
- Memory consumption.
- Number of model evaluations.
- Comparison against the existing implementation.

Do not claim performance improvements based solely on theoretical expectations.

---

## 7. Documentation Requirements

Every contribution must include appropriate documentation.

Depending on the feature, this may include:

- Explanation of the implemented functionality.
- Reference research paper.
- Mathematical formulation.
- Configuration options and their defaults.
- Available command-line arguments.
- Compatibility considerations.
- Usage instructions.
- Testing performed.
- Performance comparisons, where relevant.

Update the README or relevant documentation when a new feature introduces user-facing functionality.

Documentation should describe the actual implementation, not functionality that is planned but not yet available.

---

## 8. Pull Request Guidelines

Before submitting a pull request, ensure that the implementation satisfies the following checklist.

### Implementation

- [ ] The feature follows the existing project architecture.
- [ ] The implementation is modular and reusable.
- [ ] Existing components are reused wherever possible.
- [ ] No unnecessary duplicate pipeline or script has been introduced.
- [ ] The implementation follows the intended mathematical formulation.
- [ ] Numerical stability and relevant edge cases have been considered.

### Integration

- [ ] The feature is integrated into the existing pipeline.
- [ ] Relevant configuration options have been added.
- [ ] Relevant command-line arguments have been added.
- [ ] Command-line arguments correctly override configuration values.
- [ ] Default behavior is preserved.
- [ ] Existing functionality remains unaffected.
- [ ] Checkpoint compatibility has been considered.

### Testing

- [ ] Unit tests have been added or updated.
- [ ] Integration tests have been performed.
- [ ] The new functionality has been tested independently.
- [ ] The new functionality has been tested through the existing pipeline.
- [ ] Invalid configurations and relevant edge cases have been tested.
- [ ] Existing functionality has been regression-tested.
- [ ] No known errors or failing tests remain.

### Documentation

- [ ] Relevant documentation has been updated.
- [ ] Configuration options are documented.
- [ ] Command-line arguments are documented.
- [ ] Research references are included where applicable.
- [ ] Testing results are described.

### Pull Request Description

Please include:

1. A summary of the implementation.
2. The motivation behind the contribution.
3. Relevant research papers or references.
4. A description of the architectural changes.
5. Configuration and command-line options introduced.
6. Tests performed and their results.
7. Any known limitations or compatibility considerations.
8. Sample outputs or benchmarks where applicable.

**Pull requests containing untested implementations, incomplete pipeline integration, or undocumented breaking changes may be rejected or returned for revision.**

---

## 9. Proposing New Features

For substantial changes or new research implementations, please open an issue before beginning development.

The proposal should include:

- The feature being proposed.
- Its motivation and expected benefits.
- Relevant research papers or technical references.
- How it should integrate with the existing framework.
- Required configuration and command-line options.
- Potential compatibility considerations.
- A proposed testing strategy.

This helps avoid duplicated work and ensures that proposed changes align with the project's architecture.

Please wait for discussion before making major architectural changes.

---

## 10. Final Objective

The long-term goal of this project is to build a **modern DDPM implementation that brings together established diffusion techniques and newer research improvements within one unified PyTorch framework**.

The framework should progressively support:

- Multiple noise scheduling strategies.
- Different reverse-process variance formulations.
- Modern diffusion loss functions.
- Improved training strategies.
- EMA and other optimization techniques.
- Conditional and unconditional generation.
- Multiple sampling algorithms.
- Pixel-space and latent-space diffusion.
- Reproducible training and inference.
- Extensible model architectures.

Each contribution should move the project toward this objective.

The focus is not simply to implement more algorithms. It is to make them work together through a consistent, configurable, modular, and well-tested system.

**Every implementation must be seamlessly integrated into the existing pipeline, accessible through the appropriate configuration and command-line arguments, and tested before a pull request is submitted.**

Thank you for contributing.
