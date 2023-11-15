SHELL := /bin/bash

# Paths
PROJECT_DIR := $(shell pwd)

.PHONY: all
all: rust

.PHONY: rust
rust:
	@echo "Building Rust dependencies..."
	cargo build --release

# Run the Rust executable
.PHONY: run
run:
	cargo run --release $(PROJECT_DIR)/model/traced_model.pt

.PHONY: doc
doc:
	cargo doc --no-deps --open

.PHONY: wasm
wasm:
	wasm-pack build