use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CoreError {
    NotFitted {
        hint: &'static str,
    },
    LengthMismatch {
        left_name: &'static str,
        left: usize,
        right_name: &'static str,
        right: usize,
    },
    EmptyInput {
        what: &'static str,
    },
    DistinctNodesRequired {
        what: &'static str,
    },
    #[allow(dead_code)]
    SingularSystem {
        context: &'static str,
        pivot_index: Option<usize>,
    },
    Message(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::NotFitted { hint } => write!(f, "Interpolator not fitted. {}", hint),
            CoreError::LengthMismatch {
                left_name,
                left,
                right_name,
                right,
            } => write!(
                f,
                "{} and {} must have the same length ({} vs {})",
                left_name, right_name, left, right
            ),
            CoreError::EmptyInput { what } => write!(f, "{} cannot be empty", what),
            CoreError::DistinctNodesRequired { what } => {
                write!(f, "{} must be distinct", what)
            }
            CoreError::SingularSystem {
                context,
                pivot_index,
            } => {
                if let Some(k) = pivot_index {
                    write!(
                        f,
                        "{} is singular or ill-conditioned near pivot {}. Try fewer near-duplicate x values or adjust kernel/epsilon.",
                        context, k
                    )
                } else {
                    write!(
                        f,
                        "{} is singular or ill-conditioned. Try fewer near-duplicate x values or adjust kernel/epsilon.",
                        context
                    )
                }
            }
            CoreError::Message(msg) => write!(f, "{}", msg),
        }
    }
}

impl From<CoreError> for String {
    fn from(value: CoreError) -> Self {
        value.to_string()
    }
}
