//! # Example of how to use a PyTorch exported script module in Rust
//!
//! ## Metadata
//!
//! - **Author:** Jan-Piotraschke
//! - **Date:** 2023-Nov-15
//! - **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
//!
//! ## Current Status
//!
//! - **Bugs:** None known at this time.
//! - **Todo:** Further development tasks to be determined.


// use tch::{jit, Tensor, Device, Kind};
// use std::error::Error;
// use std::fs::File;
// use std::io::{Write, BufWriter};
// use plotters::prelude::*;

/// Run the PyTorch model and save the output to a CSV file.
// fn main() -> Result<(), Box<dyn Error>> {
//     let args: Vec<String> = std::env::args().collect();
//     if args.len() != 2 {
//         eprintln!("usage: example-app <path-to-exported-script-module>");
//         std::process::exit(1);
//     }

//     let module_path = &args[1];
//     let module = jit::CModule::load(module_path)?;

//     let num_steps = 40000;
//     let mut data = Vec::with_capacity(num_steps);
//     for i in 0..num_steps {
//         data.push(i as f32 * 0.001);
//     }

//     // Create an input tensor.
//     let input_tensor = Tensor::from_slice(&data).view([num_steps as i64, 1]);
//     // input_tensor.print(); // print the tensor
//     println!("Input: {:?}", &input_tensor);

//     let input_ivalue = tch::IValue::Tensor(input_tensor);
//     let output_ivalue = module.forward_is(&[input_ivalue])?;
//     let output = if let tch::IValue::Tensor(output) = output_ivalue {
//         output
//     } else {
//         return Err("Expected Tensor".into());
//     };

//     let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
//     println!("Output: {:?}", &output);

//     let mut file = BufWriter::new(File::create("output.csv")?);

//     let sizes = output.size();
//     let num_rows = sizes[0] as usize;
//     let num_cols = sizes[1] as usize;

//     for i in 0..num_rows {
//         for j in 0..num_cols {
//             write!(file, "{:.6}", output.double_value(&[i as i64, j as i64]))?;
//             if j != num_cols - 1 {
//                 write!(file, ",")?;
//             }
//         }
//         writeln!(file)?;
//     }

//     println!("ok");
//     Ok(())
// }


use tch::{jit, Tensor, Device, Kind};
use std::error::Error;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: example-app <path-to-exported-script-module>");
        std::process::exit(1);
    }

    let module_path = &args[1];
    let module = jit::CModule::load(module_path)?;

    let num_steps = 40000;
    let mut data = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        data.push(i as f32 * 0.001);
    }

    let input_tensor = Tensor::from_slice(&data).view([num_steps as i64, 1]);
    println!("Input: {:?}", &input_tensor);

    let input_ivalue = tch::IValue::Tensor(input_tensor);
    let output_ivalue = module.forward_is(&[input_ivalue])?;
    let output = if let tch::IValue::Tensor(output) = output_ivalue {
        output
    } else {
        return Err("Expected Tensor".into());
    };

    let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
    println!("Output: {:?}", &output);

    // Extracting data from the output tensor
    let output_iter = output.iter::<f32>()?;
    // Assuming output is a 1D tensor with the same size as `data`
    let output_data: Vec<f32> = (0..output.size()[0])
        .map(|i| output.double_value(&[i as i64]) as f32)
        .collect();

    // Plotting
    let root = BitMapBackend::new("output_plot.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Output Data Plot", ("sans-serif", 40))
        .margin(5)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0f32..40000f32, 
            output_data.iter().cloned().fold(f32::INFINITY, f32::min)..output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        )?;

    chart.configure_mesh().draw()?;

    let plot_data: Vec<(f32, f32)> = data.iter().cloned().zip(output_data.into_iter()).collect();

    chart.draw_series(LineSeries::new(plot_data, &RED))?;

    root.present()?;
    println!("Plot saved as output_plot.png");
    Ok(())
}
