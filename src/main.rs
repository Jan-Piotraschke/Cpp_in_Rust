// use tch::{kind, Tensor};

// fn grad_example() {
//     let mut x = Tensor::from(2.0f32)
//         .to_device(tch::Device::Mps)
//         .set_requires_grad(true);
//     let y = &x * &x + &x + 36;
//     println!("y {}", y.double_value(&[]));

//     x.zero_grad();
//     y.backward();

//     let dy_over_dx = x.grad();
//     println!("dy/dx {}", dy_over_dx.double_value(&[]))
// }

// fn main() {
//     let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
//     t.print(); // works on CPU tensors

//     println!("t(cpu) {:?}", &t);
//     println!("t device: {:?}", &t.device());
//     let t = Tensor::randn([5, 4], kind::FLOAT_CPU).to_device(tch::Device::Mps);
//     t.print();
//     println!("t(mps) {:?}", &t);
//     println!("t device: {:?}", &t.device());

//     grad_example();
// }

use tch::{jit, Tensor, Device, Kind};
use std::error::Error;
use std::fs::File;
use std::io::{Write, BufWriter};

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

    // let input_tensor = Tensor::of_slice(&data).view([num_steps, 1]);
    let input_tensor = Tensor::from_slice(&data).view([num_steps as i64, 1]);

    let input_ivalue = tch::IValue::Tensor(input_tensor);
    let output_ivalue = module.forward_is(&[input_ivalue])?;
    let output = if let tch::IValue::Tensor(output) = output_ivalue {
        output
    } else {
        return Err("Expected Tensor".into());
    };
    
    let mut file = BufWriter::new(File::create("output.csv")?);

    let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
    let sizes = output.size();
    let num_rows = sizes[0] as usize;
    let num_cols = sizes[1] as usize;

    for i in 0..num_rows {
        for j in 0..num_cols {
            write!(file, "{}", output.double_value(&[i as i64, j as i64]))?;
            if j != num_cols - 1 {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }

    println!("ok");
    Ok(())
}
