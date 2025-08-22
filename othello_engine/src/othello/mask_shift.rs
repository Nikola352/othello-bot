use crate::othello::mask_shift::Direction::{East, North, NorthEast, NorthWest, South, SouthEast, SouthWest, West};

pub const TOP_ROW_MASK: u64 = 0x00000000000000ffu64;
pub const BOTTOM_ROW_MASK: u64 = 0xff00000000000000u64;
pub const LEFT_COL_MASK: u64 = 0x0101010101010101u64;
pub const RIGHT_COL_MASK: u64 = 0x8080808080808080u64;
pub const MAIN_DIAGONAL_MASK: u64 = 0x8040201008040201u64;
pub const MINOR_DIAGONAL_MASK: u64 = 0x0102040810204080u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

pub const DIRECTIONS: [Direction; 8] = [North, South, East, West, NorthEast, NorthWest, SouthEast, SouthWest];

const fn shift_amount(direction: Direction) -> i32 {
    match direction {
        North => 8,
        South => 8,
        East => 1,
        West => 1,
        NorthEast => 9,
        NorthWest => 7,
        SouthEast => 7,
        SouthWest => 9,
    }
}

const fn is_left_shift(direction: Direction) -> bool {
    match direction {
        North => false,
        South => true,
        East => false,
        West => true,
        NorthEast => false,
        NorthWest => false,
        SouthEast => true,
        SouthWest => true,
    }
}

const fn shift_extra_bitmask(direction: Direction) -> u64 {
    match direction {
        North => BOTTOM_ROW_MASK,
        South => TOP_ROW_MASK,
        East => LEFT_COL_MASK,
        West => RIGHT_COL_MASK,
        NorthEast => BOTTOM_ROW_MASK | LEFT_COL_MASK,
        NorthWest => BOTTOM_ROW_MASK | RIGHT_COL_MASK,
        SouthEast => TOP_ROW_MASK | LEFT_COL_MASK,
        SouthWest => TOP_ROW_MASK | RIGHT_COL_MASK,
    }
}

pub fn shift(mask: u64, direction: Direction) -> u64 {
    let shift_by = shift_amount(direction);
    let extra_bitmask = !shift_extra_bitmask(direction);
    if is_left_shift(direction) {
        (mask << shift_by) & extra_bitmask
    } else {
        (mask >> shift_by) & extra_bitmask
    }
}
