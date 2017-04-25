## Translating Neuralese

This paper contains code and experiments for our ACL 2017 paper:

> J Andreas, A Dragan and D Klein. _Translating Neuralese_. 
> ACL 2017. https://arxiv.org/abs/1704.06960

### Getting the data

#### Driving

The data we collected for the driving game can be found in `data/drive`.
The game server itself is located in `src/server`.

#### Colors

Data comes from

> B McMahan and M Stone. _A Bayesian Model of Grounded Color
> Semantics_. TACL 2017.

    cd data/color
    wget http://paul.rutgers.edu/~bcm84/rugstk_v1.0.tar.gz
    tar xzf rugstk_v1.0.tar.gz 
    mv rugstk_v1/data/munroecorpus/* .
    rm -r rugstk_v1


#### Birds

Data comes from:

> P Welinder, S Branson, T Mita, C Wah, F Schroff, S Belongie and 
> P Perona. _Caltech-UCSD Birds 200_. Technical Report CNS-TR-2010-001,
> California Institute of Technology.

> S Reed, Z Akata, H Lee, and B Schiele. _Learning deep representations of
> fine-grained visual descriptions_. CVPR 2016.

(I am trying to figure out where the file of annotations I have came from
and whether it is public. In the meantime, contact me directly at
jda@cs.berkeley.edu if you want to run bird experiments.)

### Running the experiments

In the root of the project, first:

    mkdir experiments

To run an experiment, do:

    ./run.sh <config>

where `<config>` is the configuration file for the experiment in question
(located in the `configs` directory). Logs will be written to the appropriate
experiment dir. In general you probably want to run a `train_X` script followed
by an `eval_X` script.

#### Evaluation

This will appear in the log file when evaluation is turned on in
the config. The belief-matching criterion from our paper is called `rkl`, while
the translation criterion is called `dot`. Lines of the form [x,y:model] show
results of the pragmatic evaluation: the first number is reward and the second
is task completion. Lines of the form [x-y] show results of the semantic
evaluation.

#### Visualization

You probably also want to see what the system is saying!
Evaluation runs will create a file called `vis.json`. If you place this file in
the same directory as `vis.html` (located in the project root), you can look at
the computed beliefs associated with both natural language and neuralese
messages.
