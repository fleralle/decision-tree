# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np


class DTree:
    """ Decision Tree with weights"""

    def __init__(self):
        """ Constructor """

    def read_data(self,filename):
        fid = open(filename,"r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))
        fid.close()

        self.featureNames = data[0]
        self.featureNames = self.featureNames[:-1]
        data = data[1:]
        self.classes = []
        for d in range(len(data)):
            self.classes.append(data[d][-1])
            data[d] = data[d][:-1]

        return data,self.classes,self.featureNames

    def classify(self,tree,datapoint):

        if type(tree) == type("string"):
            # Have reached a leaf
            return tree
        else:
            a = tree.keys()[0]
            for i in range(len(self.featureNames)):
                if self.featureNames[i]==a:
                    break

            try:
                t = tree[a][datapoint[i]]
                return self.classify(t,datapoint)
            except:
                return None

    def classifyAll(self,tree,data):
        results = []
        for i in range(len(data)):
            results.append(self.classify(tree,data[i]))
        return results

    def make_tree(self,data,weights,classes,featureNames,maxlevel=-1,level=0):

        nData = len(data)
        nFeatures = len(data[0])

        try:
            self.featureNames
        except:
            self.featureNames = featureNames

        # List the possible classes
        newClasses = []
        for aclass in classes:
            if newClasses.count(aclass)==0:
                newClasses.append(aclass)

        # Compute the default class (and total entropy)
        frequency = np.zeros(len(newClasses))

        totalGini = 0
        index = 0
        for aclass in newClasses:
            frequency[index] = classes.count(aclass)
            totalGini += (float(frequency[index])/nData)**2
            index += 1

        totalGini = 1 - totalGini
        default = classes[np.argmax(frequency)]

        if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
            # Have reached an empty branch
            return default
        elif classes.count(classes[0]) == nData:
            # Only 1 class remains
            return classes[0]
        else:

            # Choose which feature is best
            #print totalGini
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                g = self.calc_info_gain(data,weights,classes,feature)
                gain[feature] = totalGini - g
            #print "gain", gain

            bestFeature = np.argmin(gain)
            #print bestFeature
            tree = {featureNames[bestFeature]:{}}

            # List the values that bestFeature can take
            values = []
            for datapoint in data:
                if values.count(datapoint[bestFeature])==0:
                    values.append(datapoint[bestFeature])

            for value in values:
                # Find the datapoints with each feature value
                newData = []
                newWeights = []
                newClasses = []
                index = 0
                for datapoint in data:
                    if datapoint[bestFeature]==value:
                        if bestFeature==0:
                            newdatapoint = datapoint[1:]
                            newweight = weights[1:]
                            newNames = featureNames[1:]
                        elif bestFeature==nFeatures:
                            newdatapoint = datapoint[:-1]
                            newweight = weights[:-1]
                            newNames = featureNames[:-1]
                        else:
                            newdatapoint = datapoint[:bestFeature]
                            newdatapoint.extend(datapoint[bestFeature+1:])
                            newweight = weights[:bestFeature]
                            newweight = np.concatenate((newweight,weights[bestFeature+1:]))
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature+1:])
                        newData.append(newdatapoint)
                        newWeights = np.concatenate((newWeights,newweight))
                        newClasses.append(classes[index])
                    index += 1

                # Now recurse to the next level
                subtree = self.make_tree(newData,newWeights,newClasses,newNames,maxlevel,level+1)

                # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree

            return tree

    def printTree(self,tree,str):
            if type(tree) == dict:
                print str, tree.keys()[0]
                for item in tree.values()[0].keys():
                        print str, item
                        self.printTree(tree.values()[0][item], str + "\t")
            else:
                print str, "\t->\t", tree

    def calc_info_gain(self,data,weights,classes,feature,maxlevel=-1,level=0):

        gain = 0
        nData = len(data)

        try:
            self.featureNames
        except:
            self.featureNames = featureNames

        # List the values that feature can take
        values = []
        valueweight = np.array([],dtype=float)
        counter = 0
        for datapoint in data:
            if values.count(datapoint[feature])==0:
                values.append(datapoint[feature])
                if np.size(valueweight) == 0:
                    valueweight = np.array([weights[counter]])
                else:
                    valueweight = np.concatenate((valueweight,np.array([weights[counter]])))
            else:
                ind = values.index(datapoint[feature])
                valueweight[ind] += weights[counter]
            counter += 1
        valueweight /= sum(valueweight)
        #print "v",valueweight
        featureCounts = np.zeros(len(values))
        gini = np.zeros(len(values))
        valueIndex = 0
        # Find where those values appear in data[feature] and the corresponding class
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature]==value:
                    featureCounts[valueIndex]+=1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1

            # Get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass)==0:
                    classValues.append(aclass)

            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex]+=1
                classIndex += 1

            for classIndex in range(len(classValues)):
                gini[valueIndex] += (float(classCounts[classIndex])/sum(classCounts))**2

            gain = gain + float(featureCounts[valueIndex])/nData * gini[valueIndex] * valueweight[valueIndex]
            valueIndex += 1
        return 1-gain
