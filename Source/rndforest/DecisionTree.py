class DecisionTree(object):
    """Generic decision tree node.
    The field 'rule' in each node corresponds to a given rule of the
    decision tree node. It consists of a list of 3-tuples that defines each
    rule. The 3-tuples values depends if we have a numerical or categorical attribute:
        -Numerical: 
            (attr,operator,value): -attr: Name of the attribute in the dataframe.
                                   -operator: can be '<' or '>='.
                                   -value: numerical value of the split.
        -Categorical:
            (attr,'in',array):     -attr: Name of the attribute in the dataframe.
                                   -array: array of values of the attribute.
    If the node has a non-void "class_atr" field, means that it is a leave.
    """
    def __init__(self, rule='*', children=None, class_atr=None):
        self.rule = rule
        self.children = []
        self.class_atr = class_atr
        if children is not None:
            for child in children:
                self.add_child(child)
                
    def print_tree(self, depth=0):
        str_format = ' '*4*depth+str(depth)+":"
        if depth==0:
            str_format += self.rule
            print(str_format)
            for child in self.children:
                child.print_tree(depth=depth+1)
        else:
            if self.class_atr is not None:
                str_format += " -> "+str(self.class_atr)
                print(str_format)
            else:
                for index, rl in enumerate(self.rule):
                    if index==0:
                        str_format += "If "+str(rl[0])+" "+str(rl[1])+" "+str(rl[2])
                    else:
                        str_format += " AND "+str(rl[0])+" "+str(rl[1])+" "+str(rl[2])
                str_format += ":"
                print(str_format)
                for child in self.children:
                    child.print_tree(depth+1)
    
    def __repr__(self):
        return self.rule
    
    def add_child(self, node):
        assert isinstance(node, DecisionTree)
        self.children.append(node)
        
    def predict_instance(self,row_instance):
        if self.class_atr is not None:
            return self.class_atr
        elif self.rule=="*":
            for child in self.children:
                pred = child.predict_instance(row_instance)
                if pred is not None:
                    return pred
            return None
        else:
            might_be_predicted = True
            for attr,rel,val in self.rule:
                if rel == "in":
                    if row_instance[attr] not in val:
                        might_be_predicted = False
                else:
                    if rel=="<":
                        if row_instance[attr] >= val:
                            might_be_predicted = False
                    else:
                        if row_instance[attr] < val:
                            might_be_predicted = False
            if might_be_predicted:
                for child in self.children:
                    pred = child.predict_instance(row_instance)
                    if pred is not None:
                        return pred
                return None
            else:
                return None
            
    def get_importance(self):
        if self.class_atr is not None:
            return []
        elif self.rule=="*":
            lst = []
            for child in self.children:
                lst.extend(child.get_importance())
            return lst
        else:
            lst = []
            for attr,rel,val in self.rule:
                lst.append(attr)
            for child in self.children:
                lst.extend(child.get_importance())
            return lst