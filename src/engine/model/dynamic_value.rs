#[derive(Debug, Clone, PartialEq)]
pub enum DynamicValue {
    Null,
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Str(String),
}

/// A value that may be overridden each frame by a SceneScript.
/// `script` holds the JS source; `value` is the static default used until
/// a JS runtime evaluates it.
#[derive(Debug, Clone)]
pub struct AnimatedValue {
    pub value: DynamicValue,
    pub script: Option<String>,
}

impl AnimatedValue {
    pub fn static_val(value: DynamicValue) -> Self {
        AnimatedValue {
            value,
            script: None,
        }
    }

    pub fn scripted(value: DynamicValue, script: String) -> Self {
        AnimatedValue {
            value,
            script: Some(script),
        }
    }

    pub fn is_animated(&self) -> bool {
        self.script.is_some()
    }

    pub fn as_float(&self) -> Option<f32> {
        match &self.value {
            DynamicValue::Float(f) => Some(*f),
            DynamicValue::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f32; 3]> {
        match &self.value {
            DynamicValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// The value's components, whatever its width — vec2 params (audiobounds,
    /// texture scales, ...) are just as real as vec3 colors.
    pub fn components(&self) -> Option<Vec<f32>> {
        match &self.value {
            DynamicValue::Vec2(v) => Some(v.to_vec()),
            DynamicValue::Vec3(v) => Some(v.to_vec()),
            DynamicValue::Vec4(v) => Some(v.to_vec()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match &self.value {
            DynamicValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            DynamicValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl From<f32> for AnimatedValue {
    fn from(v: f32) -> Self {
        Self::static_val(DynamicValue::Float(v))
    }
}

impl From<bool> for AnimatedValue {
    fn from(v: bool) -> Self {
        Self::static_val(DynamicValue::Bool(v))
    }
}

impl From<[f32; 3]> for AnimatedValue {
    fn from(v: [f32; 3]) -> Self {
        Self::static_val(DynamicValue::Vec3(v))
    }
}

impl From<String> for AnimatedValue {
    fn from(v: String) -> Self {
        Self::static_val(DynamicValue::Str(v))
    }
}

impl Default for DynamicValue {
    fn default() -> Self {
        DynamicValue::Null
    }
}

impl Default for AnimatedValue {
    fn default() -> Self {
        Self::static_val(DynamicValue::Null)
    }
}
