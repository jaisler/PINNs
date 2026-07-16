# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added validation and test to the graph neural network model (#24)
- Added software development related files (#21)
- Added initial graph neural network (#20)
- Added selection of hidden layers to apply dropout (#17)
- Added exception for metrics (#27)
- Added postprocessing tool (#28)
- Added plots for traning, validation and test datasets (#31)

### Changed
- Created a dedicated configuration directory (#19)
- Refactored the network classes using a common `BaseNetwork` (#18)
- Improved GNN architecture (#26)
- Refactored input and output handling in the main workflow (#29)
- Refactored network creation in the main workflow (#30)
- Refactored dataset splitting (#33)
- Refactored data sampling (#34)
- Refactored the main workflow (#35)
- Refactored the training pipeline (#36)
- Cleaned up the main workflow (#37)

### Removed
- Removed collocation dataset definition (#32)

### Fixed
- Fixed plot scale for the scaled absolute error plot (#38)
- Fixed collocation points conditional for plotting (#39)
