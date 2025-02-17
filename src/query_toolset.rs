use glam::*;

use crate::{CameraBuffer, QueryPod, QuerySelectionOp, QueryTexture, QueryTextureTool, QueryTool};

/// The query toolset tool.
///
/// This requires the `query-toolset` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryToolsetTool {
    /// The rectangle tool.
    Rect,

    /// The brush tool.
    Brush,
}

/// The query toolset state.
///
/// This requires the `query-toolset` feature.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryToolsetUsedTool {
    /// Using the normal query tool.
    QueryTool,

    /// Using the query texture tool.
    QueryTextureTool { selection_op: QuerySelectionOp },
}

/// The query toolset.
///
/// This contains the query tool and the query texture tool, allowing user to switch between them.
///
/// This requires the `query-toolset` feature.
#[derive(Debug)]
pub struct QueryToolset {
    /// The state.
    state: Option<(QueryToolsetUsedTool, QueryToolsetTool)>,

    /// Query maintained for texture.
    query_pod: QueryPod,
    /// Whether the texture is being used.
    use_texture: bool,
    /// The brush radius.
    brush_radius: u32,

    /// The query tool.
    query_tool: QueryTool,
    /// The query texture tool.
    query_texture_tool: QueryTextureTool,
}

impl QueryToolset {
    /// Create a new toolset.
    pub fn new(device: &wgpu::Device, query_texture: &QueryTexture, camera: &CameraBuffer) -> Self {
        Self {
            state: None,

            query_pod: QueryPod::none(),
            use_texture: false,
            brush_radius: 50,

            query_tool: QueryTool::new(),
            query_texture_tool: QueryTextureTool::new(device, query_texture, camera),
        }
    }

    /// Set whether the texture should be used.
    pub fn set_use_texture(&mut self, use_texture: bool) {
        self.use_texture = use_texture;
    }

    /// Check if the texture is being used.
    pub fn use_texture(&self) -> bool {
        self.use_texture
    }

    /// Get the query.
    ///
    /// Under the hood, it may modify its query since the last [`QueryToolset::end`] may left it in
    /// an invalid state.
    pub fn query(&self) -> &QueryPod {
        match self.use_texture() {
            false => self.query_tool.query(),
            true => &self.query_pod,
        }
    }

    /// Get the state.
    pub fn state(&self) -> Option<&(QueryToolsetUsedTool, QueryToolsetTool)> {
        self.state.as_ref()
    }

    /// Get the brush radius.
    pub fn brush_radius(&self) -> u32 {
        self.brush_radius
    }

    /// Start a query.
    pub fn start(
        &mut self,
        tool: QueryToolsetTool,
        selection_op: QuerySelectionOp,
        pos: Vec2,
    ) -> Option<&QueryPod> {
        if self.state.is_some() {
            return None;
        }

        let used_tool = match &mut self.use_texture {
            false => QueryToolsetUsedTool::QueryTool,
            true => {
                self.query_pod = QueryPod::none();
                QueryToolsetUsedTool::QueryTextureTool { selection_op }
            }
        };

        self.state = Some((used_tool, tool));

        match tool {
            QueryToolsetTool::Rect => match self.use_texture() {
                false => self.query_tool.start_rect(selection_op, pos),
                true => self.query_texture_tool.start_rect(pos),
            },
            QueryToolsetTool::Brush => match self.use_texture() {
                false => self
                    .query_tool
                    .start_brush(selection_op, self.brush_radius, pos),
                true => self.query_texture_tool.start_brush(self.brush_radius, pos),
            },
        }
        .ok()
    }

    /// Update the position.
    pub fn update_pos(&mut self, pos: Vec2) -> Option<&QueryPod> {
        match self.state {
            Some((QueryToolsetUsedTool::QueryTool, ..)) => self.query_tool.update_pos(pos).ok(),
            Some((QueryToolsetUsedTool::QueryTextureTool { .. }, ..)) => {
                match self.query_texture_tool.update_pos(pos) {
                    Ok(_) => Some(&self.query_pod),
                    Err(_) => None,
                }
            }
            None => None,
        }
    }

    /// Update the brush radius.
    pub fn update_brush_radius(&mut self, radius: u32) -> Option<&QueryPod> {
        self.brush_radius = radius;

        match self.state {
            Some((QueryToolsetUsedTool::QueryTool, QueryToolsetTool::Brush)) => {
                self.query_tool.update_brush_radius(radius).ok()
            }
            Some((QueryToolsetUsedTool::QueryTextureTool { .. }, QueryToolsetTool::Brush)) => {
                match self.query_texture_tool.update_brush_radius(radius) {
                    Ok(_) => Some(&self.query_pod),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    /// End the query.
    pub fn end(&mut self) -> Option<&QueryPod> {
        let result = match self.state {
            Some((QueryToolsetUsedTool::QueryTool, ..)) => self.query_tool.end().ok(),
            Some((QueryToolsetUsedTool::QueryTextureTool { selection_op }, ..)) => {
                match self.query_texture_tool.end() {
                    Ok(_) => {
                        self.query_pod = QueryPod::texture().with_selection_op(selection_op);
                        Some(&self.query_pod)
                    }
                    Err(_) => None,
                }
            }
            _ => None,
        };

        self.state = None;

        result
    }

    /// Render the query.
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        query_texture: &QueryTexture,
    ) {
        if self.use_texture() {
            if self.state.is_none() {
                self.query_pod = QueryPod::none();
            }

            self.query_texture_tool
                .render(queue, encoder, query_texture);
        }
    }
}
