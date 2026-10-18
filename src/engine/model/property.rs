// Agent output (qwen3-coder) — fixed: replaced hallucinated AnimatedValue::Float(v) etc.
// with the real AnimatedValue::static_val(DynamicValue::*) constructors.
// Also replaced AnimatedValue::Color{} with DynamicValue::Vec3.
use super::dynamic_value::{AnimatedValue, DynamicValue};

#[derive(Debug, Clone)]
pub enum PropertyKind {
    Slider { min: f32, max: f32, step: f32 },
    Color,
    Bool,
    Combo { options: Vec<String> },
    Text,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Property {
    pub key: String,
    pub text: String,
    pub value: AnimatedValue,
    pub kind: PropertyKind,
}

#[derive(Debug, Clone)]
pub struct UserSetting {
    pub property: Property,
    pub user_value: Option<AnimatedValue>,
}

impl Property {
    pub fn from_json(key: &str, v: &serde_json::Value) -> Option<Self> {
        let text = v.get("text")?.as_str()?.to_string();
        let kind = match v.get("type")?.as_str()? {
            "slider" => {
                let min = v.get("min")?.as_f64()? as f32;
                let max = v.get("max")?.as_f64()? as f32;
                let step = v.get("step").and_then(|s| s.as_f64()).unwrap_or(0.01) as f32;
                PropertyKind::Slider { min, max, step }
            }
            "color" => PropertyKind::Color,
            "bool" => PropertyKind::Bool,
            "combo" => {
                let options = v
                    .get("options")?
                    .as_array()?
                    .iter()
                    .map(|o| o.as_str().unwrap_or("").to_string())
                    .collect();
                PropertyKind::Combo { options }
            }
            "text" => PropertyKind::Text,
            _ => PropertyKind::Unknown,
        };

        let raw = v.get("value")?;
        let value = match &kind {
            PropertyKind::Slider { .. } => {
                AnimatedValue::static_val(DynamicValue::Float(raw.as_f64()? as f32))
            }
            PropertyKind::Color => {
                let s = raw.as_str()?;
                let parts: Vec<f32> = s
                    .split_whitespace()
                    .filter_map(|p| p.parse().ok())
                    .collect();
                let rgb = if parts.len() >= 3 {
                    [parts[0], parts[1], parts[2]]
                } else {
                    [0.0; 3]
                };
                AnimatedValue::static_val(DynamicValue::Vec3(rgb))
            }
            PropertyKind::Bool => AnimatedValue::static_val(DynamicValue::Bool(raw.as_bool()?)),
            PropertyKind::Combo { .. } => {
                AnimatedValue::static_val(DynamicValue::Int(raw.as_i64()? as i32))
            }
            PropertyKind::Text => {
                AnimatedValue::static_val(DynamicValue::Str(raw.as_str()?.to_string()))
            }
            PropertyKind::Unknown => AnimatedValue::default(),
        };

        Some(Property {
            key: key.to_string(),
            text,
            value,
            kind,
        })
    }
}

impl UserSetting {
    pub fn effective_value(&self) -> &AnimatedValue {
        self.user_value.as_ref().unwrap_or(&self.property.value)
    }

    pub fn set_user_value(&mut self, v: AnimatedValue) {
        self.user_value = Some(v);
    }
}
