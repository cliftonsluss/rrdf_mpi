use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub fn read_lines<P>(filename: P) -> Result<io::Lines<io::BufReader<File>>, io::Error>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

use std::f64::consts::PI;

/// Basic structure for handling a single trajectory frame
pub struct Frame {
    pub simbox: Vec<Vec<f64>>,
    pub data: Vec<Vec<f64>>,
    pub natoms: usize,
}

impl Frame {
    /// Contructs a new frame by declaring vectors to store the x,y,z box limits
    /// and atom data
    ///
    pub fn new(natoms: usize) -> Result<Frame, std::io::Error> {
        Ok(Self {
            simbox: vec![Vec::new(); 3],
            data: vec![Vec::new(); natoms],
            natoms,
        })
    }
    /// Methods to set the x,y,z box limits for the frame.
    pub fn add_x(&mut self, bound: String) {
        for number in bound.split_whitespace() {
            self.simbox[0].push(number.parse().unwrap());
        }
    }

    pub fn add_y(&mut self, bound: String) {
        for number in bound.split_whitespace() {
            self.simbox[1].push(number.parse().unwrap());
        }
    }

    pub fn add_z(&mut self, bound: String) {
        for number in bound.split_whitespace() {
            self.simbox[2].push(number.parse().unwrap());
        }
    }
    /// Method used to add the coordinates of a single atom to the frame structure.
    ///
    pub fn add_atom(&mut self, atom: String) {
        let mut iter = atom.split_ascii_whitespace();
        let mut index: usize = iter.next().unwrap().parse().unwrap();
        index -= 1;
        let _at = iter.next();
        let mut loc: f64;
        for (_a, b) in self.simbox.iter().zip(iter) {
            loc = b.parse::<f64>().unwrap();
            self.data[index].push(loc);
        }
    }

    pub fn ingest(&mut self, filename: &String, fnum: usize) {
        let flines: usize = self.natoms + 9;

        if let Ok(mut lines) = read_lines(filename) {
            for _i in 1..=(flines * (fnum - 1)) {
                lines.next();
            }
            for (line, ln) in lines.zip(1..=flines) {
                match ln {
                    1 | 2 | 3 | 4 | 5 | 9 => (),
                    6 => self.add_x(line.unwrap()),
                    7 => self.add_y(line.unwrap()),
                    8 => self.add_z(line.unwrap()),
                    _ => self.add_atom(line.unwrap()),
                }
            }
        }
    }
}

/// Helper function to calculate the squared L2 distance between two atoms.
/// Takes two vectors of len 3, applies the minimum image criteria and returns
/// the sum of the squared differences.
/// The squared L2 distance lets us compare to the rmax^2 distance and only
/// perform the more expensive sqrt calculation if the pair is inside the sphere
/// described by rmax.
pub fn l22(a: &Vec<f64>, b: &Vec<f64>, simbox: &Vec<Vec<f64>>) -> f64 {
    let mut result: f64 = 0.0;

    for ((a, b), c) in a.iter().zip(b.iter()).zip(simbox.iter()) {
        let mut dist = a - b;
        if dist > 0.5 {
            dist = dist - 1.0;
        }
        if dist < -0.5 {
            dist = dist + 1.0;
        }
        dist = dist * (c[1] - c[0]);
        result = result + (dist * dist);
    }
    result
}

/// Basic structure to contain a RDF and
pub struct Rdf {
    pub gvec: Vec<f64>,
    pub rvec: Vec<f64>,
    rmax2: f64,
    dr: f64,
    pub fcount: usize,
    volume: f64,
    natoms: usize,
}

impl Rdf {
    /// Create a new Rdf instance, calculating rmax2, dr, and populating the r vector rvec
    pub fn new(rmax: f64, nbins: usize, natoms: usize) -> Result<Rdf, std::io::Error> {
        let dr = rmax / (nbins as f64);
        let mut rvec: Vec<f64> = vec![0.0; nbins];
        for i in 0..nbins {
            rvec[i] = (i as f64) * dr;
        }
        Ok(Self {
            gvec: vec![0.0; nbins],
            rvec,
            rmax2: rmax * rmax,
            dr,
            fcount: 0,
            volume: 0.0,
            natoms,
        })
    }
    /// For each frame we to calculate the distance between every atom and all of its
    /// neighbors that are inside a sphere with radius rmax. These distances are then
    /// binned into bins of width dr spanning a total distance r of rmax. It is assumed
    /// that the simulation has been performed with constant volume. For this reason
    /// the volume calculated only once, from the first frame that is binned.
    pub fn bin_frame(&mut self, frame: Frame) {
        let mut dist2: f64;
        let mut i: usize = 1;
        if self.fcount == 0 {
            let x = frame.simbox[0][1] - frame.simbox[0][0];
            let y = frame.simbox[1][1] - frame.simbox[1][0];
            let z = frame.simbox[2][1] - frame.simbox[2][0];
            self.volume = x * y * z;
        }
        for atom in frame.data.iter() {
            for ca in frame.data[i..].iter() {
                dist2 = l22(atom, ca, &frame.simbox);
                if dist2 < self.rmax2 {
                    if dist2 < 0.01 {
                        //    println!("{:?}, {:?}", atom, ca);
                    }
                    self.gvec[(dist2.sqrt() / self.dr) as usize] += 1.0;
                }
            }
            i += 1;
        }
        self.fcount += 1;
    }
    /// After all the frames of interest have been binned the resulting Rdf needs to
    /// be scaled by the incremental spherical shell volume to
    pub fn scale(&mut self) -> Result<(), &'static str> {
        if self.fcount != 0 {
            let mut del_v: f64;
            let mut r1: f64;
            let mut r2: f64;
            let fac = self.volume / ((self.natoms * self.natoms * self.fcount) as f64);
            for (i, r) in (0..).zip(&self.rvec) {
                r1 = r + self.dr * 0.5;
                r2 = r - self.dr * 0.5;
                del_v = 1.0 / ((4.0 / 3.0) * PI * (r1.powi(3) - r2.powi(3)));
                self.gvec[i] = self.gvec[i] * del_v * fac * 2.0;
            }
            Ok(())
        } else {
            Err("At least one frame must be binned before scaling rdf")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    /// l22 computes the l2 distance between two atoms in the simulation box. It should
    /// apply the minimum image criteria to account for periodic boundary conditions of
    /// the simulation box. l22 should return the square of the l2 distance.
    #[test]
    fn l22_can_compute_distance() {
        let a: Vec<f64> = vec![0.0, 0.0, 0.3];
        let b: Vec<f64> = vec![0.0, 0.4, 0.0];
        let simbox: Vec<Vec<f64>> = vec![vec![0.0, 10.0], vec![0.0, 10.0], vec![0.0, 10.0]];
        let result = l22(&a, &b, &simbox);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn l22_applies_minimum_image() {
        let a: Vec<f64> = vec![0.0, 0.0, 0.8];
        let b: Vec<f64> = vec![0.0, 0.0, 0.2];
        let simbox: Vec<Vec<f64>> = vec![vec![0.0, 10.0], vec![0.0, 10.0], vec![0.0, 10.0]];
        let result = l22(&a, &b, &simbox);
        assert!((result - 16.0).abs() < 1.0e-13);
    }

    #[test]
    fn frame_new_creates_data_vector() {
        let frame = Frame::new(64).unwrap();
        assert_eq!(frame.data.len(), 64);
    }

    #[test]
    fn add_x_creates_xbound() {
        let mut frame = Frame::new(64).unwrap();
        frame.add_x(String::from("0.0 10.0\n"));
        assert_eq!(frame.simbox[0][0], 0.0);
    }

    #[test]
    fn read_lines_provides_error() {
        let lines = read_lines("not_a_file");

        let mut lines = match lines {
            Ok(lines) => assert_eq!("Pass", "Fail"),
            Err(e) => assert_eq!("Pass", "Pass"),
        };
    }

    #[test]
    fn frame_reads_box_bounds() {
        let na: usize = 128;
        let mut frame = Frame::new(na).unwrap();
        let file_path = String::from("test_data/Fe_128a_box_check.trj");
        frame.ingest(&file_path, 1);
        let xbound = frame.simbox[0][1] - frame.simbox[0][0];
        let ybound = frame.simbox[1][1] - frame.simbox[1][0];
        let zbound = frame.simbox[2][1] - frame.simbox[2][0];
        assert_eq!(xbound, 10.0);
        assert_eq!(ybound, 10.0);
        assert_eq!(zbound, 10.0);
    }

    #[test]
    fn frame_reads_atoms() {
        let na: usize = 128;
        let mut frame = Frame::new(na).unwrap();
        let file_path = String::from("test_data/Fe_128a_box_check.trj");
        frame.ingest(&file_path, 1);
        let ax = frame.data[1][0];
        let ay = frame.data[1][1];
        let az = frame.data[1][2];
        assert_eq!(ax, 0.154026);
        assert_eq!(ay, 0.156444);
        assert_eq!(az, 0.0876359);
    }
}
