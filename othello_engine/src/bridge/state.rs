use crate::othello::board::Piece::{Empty, Occupied};
use crate::othello::board::{Square, BOARD_SIZE};
use crate::othello::color::Color;
use crate::othello::state::GameState as NativeGameState;
use numpy::PyArray2;
use pyo3::prelude::*;
use Color::{Black, White};

/// Python-compatible wrapper for the native GameState
#[pyclass]
#[derive(Debug)]
pub struct PyGameState {
    inner: NativeGameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    pub fn new() -> Self {
        PyGameState {
            inner: NativeGameState::new(),
        }
    }

    /// Get legal moves as a list of tuples (row, col)
    pub fn get_legal_moves(&self) -> Vec<(u8, u8)> {
        self.inner
            .get_legal_moves()
            .iter()
            .map(square_to_tuple)
            .collect()
    }

    /// Play a move given coordinates (row, col)
    pub fn play_move(&self, row: u8, col: u8) -> PyResult<PyGameState> {
        let square = Square::new(row, col);
        let new_state = self.inner.play_move(&square);
        Ok(PyGameState { inner: new_state })
    }

    /// Skip the current turn
    pub fn skip_turn(&self) -> PyGameState {
        PyGameState {
            inner: self.inner.skip_turn(),
        }
    }

    /// Get the current score (positive = black advantage, negative = white advantage)
    pub fn get_score(&self) -> i32 {
        self.inner.get_score()
    }

    /// Check if the game is finished
    pub fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    /// Get the winner: 1 = Black, -1 = White, 0 = Draw/Game not finished
    pub fn get_winner(&self) -> i8 {
        self.inner.get_winner().map(color_to_int).unwrap_or(0)
    }

    /// Get the current player's turn: 1 = Black, -1 = White
    pub fn get_turn(&self) -> i8 {
        color_to_int(self.inner.turn.clone())
    }

    /// Get piece counts as (black_count, white_count)
    pub fn get_piece_counts(&self) -> (u32, u32) {
        self.inner.board.piece_counts()
    }

    /// Get the board state as a 2d numpy array
    /// 0 = Empty, 1 = Black, -1 = White
    pub fn get_board<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mut board_data = vec![vec![0f32; BOARD_SIZE as usize]; BOARD_SIZE as usize];
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                board_data[row as usize][col as usize] =
                    piece_to_int(&self.inner.board.get(&Square::new(row, col))) as f32
            }
        }
        Ok(PyArray2::from_vec2(py, &board_data)?)
    }

    /// Get a board mask where 1 represents a legal move and 0 everything else.
    pub fn get_legal_moves_mask<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let legal_moves = self.inner.get_legal_moves();
        let mut board = vec![vec![0f32; BOARD_SIZE as usize]; BOARD_SIZE as usize];
        for square in legal_moves {
            board[square.row as usize][square.col as usize] = 1f32;
        }
        Ok(PyArray2::from_vec2(py, &board)?)
    }

    /// String representation of the game state
    pub fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// String representation for debugging
    pub fn __repr__(&self) -> String {
        format!(
            "PyGameState(turn={}, score={})",
            self.get_turn(),
            self.get_score()
        )
    }

    pub fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone()
        }
    }
}

// Helper functions for type conversion
fn square_to_tuple(square: &Square) -> (u8, u8) {
    (square.row(), square.col())
}

fn color_to_int(color: Color) -> i8 {
    match color {
        Black => 1,
        White => -1,
    }
}

fn piece_to_int(piece: &crate::othello::board::Piece) -> i8 {
    match piece {
        Empty => 0,
        Occupied(Black) => 1,
        Occupied(White) => -1,
    }
}
