import os

def process_training_set_plot(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        trainingSet = file.readlines()  

    plots = []  

    for line in trainingSet:
        fields = line.split("\t")
        plot = fields[4]  
        plots.append(plot)

    
    with open(output_file, 'w', encoding='utf-8') as f:
        for plot in plots:
            f.write(plot)

    return plots  


#TestExample
#trainTxt_path = os.path.join('..', 'train.txt')
#output_path = os.path.join('..', 'testEmbeddings.txt')

#process_training_set_plot(trainTxt_path, output_path)

