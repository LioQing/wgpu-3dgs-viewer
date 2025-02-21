use glam::*;

use crate::{Error, QueryPod, QuerySelectionOp};

/// The query tool state.
///
/// This requires the `query-tool` feature.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryToolState {
    /// The rectangle tool state.
    Rect { start: Vec2 },

    /// The brush tool state.
    Brush,
}

/// The query tool.
///
/// It allows updating the query using the query types: rectangle and brush.
///
/// For [`QueryTexture`](crate::QueryTexture), the `query-texture-tool` feature is required.
///
/// This requires the `query-tool` feature.
#[derive(Debug)]
pub struct QueryTool {
    /// The state.
    state: Option<QueryToolState>,

    /// The query.
    query: QueryPod,
}

impl QueryTool {
    /// Create a new query tool.
    pub fn new() -> Self {
        Self {
            state: None,
            query: QueryPod::none(),
        }
    }

    /// Get the query.
    pub fn query(&self) -> &QueryPod {
        &self.query
    }

    /// Get the state.
    pub fn state(&self) -> Option<&QueryToolState> {
        self.state.as_ref()
    }

    /// Start a [`QueryType::Rect`](crate::QueryType::Rect) query.
    pub fn start_rect(
        &mut self,
        selection_op: QuerySelectionOp,
        pos: Vec2,
    ) -> Result<&QueryPod, Error> {
        if self.state.is_some() {
            return Err(Error::QueryToolAlreadyInUse);
        }

        self.query = QueryPod::rect(pos, pos).with_selection_op(selection_op);
        self.state = Some(QueryToolState::Rect { start: pos });

        Ok(&self.query)
    }

    /// Start a [`QueryType::Brush`](crate::QueryType::Brush) query.
    pub fn start_brush(
        &mut self,
        selection_op: QuerySelectionOp,
        radius: u32,
        pos: Vec2,
    ) -> Result<&QueryPod, Error> {
        if self.state.is_some() {
            return Err(Error::QueryToolAlreadyInUse);
        }

        self.query = QueryPod::brush(radius, pos, pos).with_selection_op(selection_op);
        self.state = Some(QueryToolState::Brush);

        Ok(&self.query)
    }

    /// Update the position.
    pub fn update_pos(&mut self, pos: Vec2) -> Result<&QueryPod, Error> {
        match self.state {
            Some(QueryToolState::Rect { start }) => {
                let selection_op = match self.query.query_selection_op() {
                    QuerySelectionOp::Add => QuerySelectionOp::Add,
                    QuerySelectionOp::Remove => QuerySelectionOp::Remove,
                    QuerySelectionOp::None | QuerySelectionOp::Set => QuerySelectionOp::Set,
                };

                let top_left = start.min(pos);
                let bottom_right = start.max(pos);

                self.query = QueryPod::rect(top_left, bottom_right).with_selection_op(selection_op);

                Ok(&self.query)
            }
            Some(QueryToolState::Brush) => {
                let selection_op = match self.query.query_selection_op() {
                    QuerySelectionOp::Add | QuerySelectionOp::Set => QuerySelectionOp::Add,
                    QuerySelectionOp::Remove => QuerySelectionOp::Remove,
                    QuerySelectionOp::None => QuerySelectionOp::Set,
                };

                self.query = QueryPod::brush(
                    self.query.as_brush().radius(),
                    self.query.as_brush().end(),
                    pos,
                )
                .with_selection_op(selection_op);

                Ok(&self.query)
            }
            None => Err(Error::QueryToolNotInUse),
        }
    }

    /// Update the brush radius.
    pub fn update_brush_radius(&mut self, radius: u32) -> Result<&QueryPod, Error> {
        match self.state {
            Some(QueryToolState::Brush) => {
                self.query = QueryPod::brush(
                    radius,
                    self.query.as_brush().start(),
                    self.query.as_brush().end(),
                )
                .with_selection_op(self.query.query_selection_op());

                Ok(&self.query)
            }
            _ => Err(Error::QueryToolNotInUse),
        }
    }

    /// End the query.
    pub fn end(&mut self) -> Result<&QueryPod, Error> {
        if self.state.is_none() {
            return Err(Error::QueryToolNotInUse);
        }

        self.query = QueryPod::none();
        self.state = None;

        Ok(&self.query)
    }
}

impl Default for QueryTool {
    fn default() -> Self {
        Self::new()
    }
}
