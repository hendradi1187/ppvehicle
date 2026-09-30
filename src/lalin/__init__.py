"""Wrapper around PaddleDetection's PP-Vehicle pipeline: config composition,
model management and a job API.

Named `lalin` (lalu lintas), NOT `ppvehicle`: PaddleDetection ships its own
package at deploy/pipeline/ppvehicle/, and pipeline.py imports it as
`ppvehicle.vehicle_plate`. A wrapper package with that name shadows it - an
editable install registers a meta path finder, which is consulted before the
sys.path search, so upstream loses the import even when its own directory is
sys.path[0].
"""

__version__ = "0.1.0"
