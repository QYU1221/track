use crate::{
    contracts::ObservationPacket,
    efficient_config::ResourceSampler,
    error::{Error, Result},
    factorized_tracker::Tracker,
    runtime_model,
};
use serde_json::{json, Value};
use std::collections::HashSet;
pub struct Session {
    pub tracker: Tracker,
    packet: std::sync::Arc<ObservationPacket>,
    selected: Vec<i64>,
    detail: String,
}

impl Session {
    pub fn new_shared(
        packet_json: &Value,
        packet: std::sync::Arc<ObservationPacket>,
        raw: &Value,
        checkpoint: Option<&Value>,
    ) -> Result<Self> {
        let model = runtime_model::build_model_with_packet(packet_json, &packet, raw)?;
        let mut tracker = model.tracker()?;
        if let Some(checkpoint) = checkpoint {
            tracker.segment_events = crate::checkpoint::restore(
                &mut tracker.state,
                checkpoint,
                &model.binding_digest()?,
                model.input_frame_digests(),
            )?;
        }
        let requested = raw
            .get("frames")
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| Error::value("requested frames missing or repeated"))
            })
            .transpose()?;
        let selected = packet
            .frames
            .iter()
            .filter(|f| {
                requested.is_none_or(|a| a.iter().any(|v| v.as_i64() == Some(f.frame)))
                    && tracker.state.frame.is_none_or(|old| f.frame > old)
            })
            .map(|f| f.frame)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Err(Error::value("no requested frames after checkpoint"));
        }
        if let Some(a) = requested {
            let ids = a.iter().map(Value::as_i64).collect::<Vec<_>>();
            let known = packet
                .frames
                .iter()
                .map(|f| f.frame)
                .collect::<HashSet<_>>();
            if ids.iter().collect::<HashSet<_>>().len() != ids.len()
                || ids.iter().any(|x| x.is_none_or(|n| !known.contains(&n)))
            {
                return Err(Error::value("requested frames missing or repeated"));
            }
        }
        let detail = raw["reporting"]["detail"]
            .as_str()
            .ok_or_else(|| Error::value("reporting detail must be compact or full"))?
            .to_owned();
        Ok(Self {
            tracker,
            packet,
            selected,
            detail,
        })
    }
    pub fn selected_frames(&self) -> &[i64] {
        &self.selected
    }
    pub fn run<S: ResourceSampler, F: FnMut() -> S, C: FnMut(Value) -> Result<()>>(
        &mut self,
        mut sampler: F,
        mut consume: C,
    ) -> Result<Option<Value>> {
        for number in &self.selected {
            if self.tracker.state.frame.is_some_and(|old| *number <= old) {
                continue;
            }
            let frame = self
                .packet
                .frames
                .iter()
                .find(|f| f.frame == *number)
                .unwrap();
            let detections = self
                .packet
                .detections
                .iter()
                .filter(|d| d.frame == *number)
                .cloned()
                .collect::<Vec<_>>();
            let result =
                self.tracker
                    .step_with_segmentation(frame, &detections, &self.detail, &mut sampler);
            match result {
                Ok((mut result, _event)) => {
                    result["segment_id"] = json!(self.tracker.segment_events.len() + 1);
                    consume(result)?;
                }
                Err(e)
                    if ["InferenceError", "ValueError", "FloatingPointError"].contains(&e.kind) =>
                {
                    return Ok(Some(json!({
                        "frame": number,
                        "error": e.message,
                        "previous_committed_frame": self.tracker.state.frame
                    })))
                }
                Err(e) => return Err(e),
            }
        }
        Ok(None)
    }
}
