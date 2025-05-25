import evolve
import argparse
# from elevate import elevate


def main(input_dir: str|None, output_dir:str|None, batchsize:int, epochs:int, iterations:int, fake_delim:int, rows:int=6, columns:int=7):
    # elevate()
    if (input_dir==None):
        b = evolve.random_batch(6, 7, batchsize)
    else:
        b = evolve.Batch.load(rows, columns, input_dir)
    
    if (output_dir==None): output_dir=f"epoch{epochs}/"
    
    b = evolve.evolve_with_fake(b, epochs, iterations, fake_delim, random_factor=0.06)
    b.save(output_dir)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Args")
    
    
    parser.add_argument('batchsize', type=int, help='Size of one batch (epoch)')
    parser.add_argument('epochs', type=int, help='Number of epochs')
    parser.add_argument('iterations', type=int, help='Number of iterations per epoch')
    parser.add_argument('fake_delim', type=int, help='Fake NN include switch num')
    
    parser.add_argument('--indir', type=str, help='Input batch folder', default=None)
    parser.add_argument('--outdir', type=str, help='Output batch folder', default=None)
    parser.add_argument('--rows', type=int, help='Rows', default=6)
    parser.add_argument('--columns', type=int, help='Columns', default=7)
    Args = parser.parse_args()
    
    main(
        input_dir=Args.indir,
        output_dir=Args.outdir,
        batchsize=Args.batchsize,
        epochs=Args.epochs,
        iterations=Args.iterations,
        fake_delim=Args.fake_delim,
        rows=Args.rows,
        columns=Args.columns
    )