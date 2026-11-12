#![recursion_limit = "256"]

pub mod contracts;
pub mod error;
pub mod existence;
pub mod numerical_policy;
mod linalg;
mod numpy_arithmetic;
mod quadrature;

pub mod camera_model;
pub mod geometry;
pub mod gnomonic;
pub mod measurement_covariance;
pub mod observation_geometry;
pub mod exact_sources;
mod spatial_index;

pub mod data_io;
pub mod table_io;
pub mod table_stream;
pub mod exact_io;

pub mod configuration;
pub mod profile_binding;
pub mod runtime_model;

pub mod exact_numerics;
pub mod exact_factors;
pub mod exact_posterior;
pub mod efficient_config;
pub mod efficient_graph;
pub mod efficient_solver;

pub mod tracking_model;
pub mod tracking_background;
pub mod online_factors;
pub mod missed_update;
pub mod birth_update;
pub mod undetected;
pub mod factorized_groups;
pub mod conditional_feedback;
pub mod joint_update;
pub mod hypothesis_selection;
pub mod factorized_tracker;

pub mod tracking_lifecycle;
pub mod background_lifecycle;
pub mod candidates;
pub mod finish_management;
pub mod frame_commit;

pub mod tracking_marginals;
pub mod point_marginals;

pub mod input_evidence;
pub mod checkpoint;

pub mod sequence_session;
pub mod production;
pub mod cancellation;
pub mod progress;

pub mod report_ledger;
pub mod report_schema;
pub mod star_display;
pub mod report_identity;
pub mod report_records;
pub mod report_files;
pub mod report_store;
pub mod report_canvas;
pub mod report_angles;
pub mod report_plots;
pub mod reporter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError(pub &'static str);
impl ValidationError {
    pub fn code(&self) -> Option<&'static str> {
        match self.0 {
            "LOS outside declared gnomonic domain"
            | "gnomonic position outside declared domain" => Some("tangent_domain"),
            _ => None,
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for ValidationError {}
