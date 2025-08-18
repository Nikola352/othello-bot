use crate::othello::color::Color::{Black, White};
use std::ops;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opponent(&self) -> Color {
        match self {
            Black => White,
            White => Black,
        }
    }
}

impl ops::Neg for Color {
    type Output = Color;

    fn neg(self) -> Color {
        self.opponent()
    }
}
