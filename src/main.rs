use rrdf_mpi::{Frame, Rdf};
use std::env;

extern crate mpi;

use mpi::collective::SystemOperation;
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let mut na: usize = 128;
    let mut rmax: f64 = 11.0;
    let mut nbins: usize = 1100;
    let mut nframes: usize = 40;
    let root_frames: usize = nframes / (size as usize);

    root_process.broadcast_into(&mut na);
    root_process.broadcast_into(&mut rmax);
    root_process.broadcast_into(&mut nbins);
    root_process.broadcast_into(&mut nframes);

    let mut rdf = Rdf::new(rmax, nbins, na).unwrap();

    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];

    if rank == root_rank {
        for f in 1..=root_frames {
            let mut frame = Frame::new(na).unwrap();
            frame.ingest(file_path, f);
            rdf.bin_frame(frame);
        }
        let mut rdft = Rdf::new(rmax, nbins, na).unwrap();
        world.process_at_rank(root_rank).reduce_into_root(
            &rdf.gvec,
            &mut rdft.gvec,
            SystemOperation::sum(),
        );
        world.process_at_rank(root_rank).reduce_into_root(
            &rdf.fcount,
            &mut rdft.fcount,
            SystemOperation::sum(),
        );

        rdft.scale().unwrap();
        for (r, g) in rdft.rvec.iter().zip(rdft.gvec) {
            println!("{:?}, {:?}", r, g);
        }
        println!("Read {} frames", rdft.fcount);
        println!("root frames {}", root_frames);
    } else {
        let start: usize = ((rank as usize) * root_frames) + 1;
        let stop: usize = start + root_frames;
        for f in start..stop {
            let mut frame = Frame::new(na).unwrap();
            frame.ingest(file_path, f);
            rdf.bin_frame(frame);
        }
        world
            .process_at_rank(root_rank)
            .reduce_into(&mut rdf.gvec, SystemOperation::sum());
        world
            .process_at_rank(root_rank)
            .reduce_into(&rdf.fcount, SystemOperation::sum());
        println!("read {} frames at rank {}", rdf.fcount, rank);
    }
}
