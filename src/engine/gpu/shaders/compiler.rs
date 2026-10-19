use anyhow::{Context, Result};

/// Trait abstracting the GLSL → SPIR-V compilation step.
///
/// The default implementation uses `shaderc` (wraps glslangValidator as a
/// statically-linked library — no system binary required, works on Linux and macOS).
/// `NagaGlslCompiler` is a stub for a future pure-Rust fallback via naga's GLSL frontend.
pub trait GlslCompiler: Send + Sync {
    fn compile(&self, glsl: &str, stage: naga::ShaderStage, name: &str) -> Result<Vec<u8>>;
}

// ── shaderc implementation ─────────────────────────────────────────────────────

pub struct ShadercCompiler;

impl GlslCompiler for ShadercCompiler {
    fn compile(&self, glsl: &str, stage: naga::ShaderStage, name: &str) -> Result<Vec<u8>> {
        let kind = stage_to_kind(stage);

        let compiler = shaderc::Compiler::new().context("failed to create shaderc compiler")?;
        let mut options =
            shaderc::CompileOptions::new().context("failed to create shaderc options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_1 as u32,
        );
        options.set_source_language(shaderc::SourceLanguage::GLSL);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        let artifact = compiler
            .compile_into_spirv(glsl, kind, name, "main", Some(&options))
            .context("shaderc GLSL→SPIR-V compilation failed")?;

        Ok(artifact.as_binary_u8().to_vec())
    }
}

// ── naga GLSL frontend stub ───────────────────────────────────────────────────

/// Placeholder for a future pure-Rust GLSL→SPIR-V path via naga's glsl-in frontend.
/// Would eliminate the C++ shaderc build dependency at the cost of parser maturity.
pub struct NagaGlslCompiler;

impl GlslCompiler for NagaGlslCompiler {
    fn compile(&self, _glsl: &str, _stage: naga::ShaderStage, _name: &str) -> Result<Vec<u8>> {
        anyhow::bail!("naga GLSL frontend not yet implemented — use ShadercCompiler")
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

pub fn default_compiler() -> Box<dyn GlslCompiler> {
    Box::new(ShadercCompiler)
}

fn stage_to_kind(stage: naga::ShaderStage) -> shaderc::ShaderKind {
    match stage {
        naga::ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
        naga::ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
        naga::ShaderStage::Compute => shaderc::ShaderKind::Compute,
        _ => shaderc::ShaderKind::InferFromSource,
    }
}
