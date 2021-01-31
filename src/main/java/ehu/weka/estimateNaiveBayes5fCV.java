package ehu.weka;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.Random;

public class estimateNaiveBayes5fCV {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println(
                    "Helburua: emandako datuekin Naive Bayes-en kalitatearen estimazioa lortu 5-fCV eskemaren bidez eta datuei buruzko informazioa eman Argumentuak:"+"\n"
                    +"1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da."+"\n"+
                    "2. Emaitzak idazteko irteerako fitxategiaren path-a (output).");
            System.out.println("\nUsage: CSV2Arff </path/data.arff> </path/emaitzak.txt>\n");
            System.exit(0);
        }
        else{
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            datuakinprimatu(data);
            fitxategiaSortu(entrenatu(data), args[1]);
        }

    }
    private static void datuakinprimatu(Instances data) {
        System.out.println("-------------------------------------------------------------");
        System.out.println("Datu sorta honetan " + data.numInstances() + " instantzia daude");
        System.out.println("Datu sorta honetan " + data.numAttributes() + " atributu daude");
        System.out.println("Datu sorta honetan, lehenengo atributuak  " + data.numDistinctValues(0) + " balio desberdin hartu ditzake");
        System.out.println("Datu sorta honetan, azken-aurreko atributuak  " + data.attributeStats(data.numAttributes() - 2).missingCount + "missing values ditu");
        System.out.println("-------------------------------------------------------------");
    }

    private static Evaluation entrenatu(Instances data) throws Exception {
        NaiveBayes model=new NaiveBayes();
        model.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 5, new Random(1));

        return eval;
    }

    private static void fitxategiaSortu(Evaluation eval, String path) {
        java.util.Date date = new java.util.Date();
        FileWriter myWriter = null;
        try {
            myWriter = new FileWriter(path);
            myWriter.write(String.valueOf(date)+"\n");
            myWriter.write(path+"\n");
            myWriter.write(eval.toMatrixString());
            myWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
