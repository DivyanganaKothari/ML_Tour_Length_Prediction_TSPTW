class FeatureManager:
    def __init__(self):
        self.feature_sets = {

            "set1": ['Total Number of Nodes', 'MinP', 'MaxP', 'VarP', 'SumMinP', 'SumMaxP', 'MinM', 'MaxM',
                     'SumM', 'VarM', 'Mean Distance', 'Median Distance', 'Std Distance', 'Sum Distance', 'Sum of Min Distance',
                     'Sum of Max Distance', 'Percentile 25 Distance', 'Percentile 50 Distance', 'Percentile 75 Distance',
                     'Total Time Window', 'Average Time Window', 'Standard Deviation of Time Window',
                     'Average Earliest Time', 'Standard Deviation Earliest Time', 'Average Latest Time',
                     'Standard Deviation Latest Time', 'Mean Time Window', 'Sum of Distance to Depot',
                     'Average Distance to Depot', 'Maximum Distance to Depot',
                     'Minimum Distance to Depot', 'Standard Deviation of Distance to Depot', 'Tour Length',
                     'Min Earliest Time', 'Max Earliest Time', 'Min Latest Time', 'Max Latest Time'],
            "set2": ['MinP', 'MaxP', 'MinM', 'MaxM', 'SumM', 'VarP', 'Std Distance', 'Sum of Min Distance',
                     'Sum of Max Distance', 'Total Time Window', 'Sum of Distance to Depot'
                    ,'Maximum Distance to Depot', 'Tour Length', 'Min Earliest Time', 'Max Earliest Time',
                     'Min Latest Time', 'Max Latest Time', 'Standard Deviation of Distance to Depot'],
            "set3": ['Max Earliest Time', 'Min Earliest Time', 'MaxP', 'Sum of Max Distance', 'MinM', 'Min Latest Time',
                     'SumM', 'Std Distance', 'Standard Deviation of Distance to Depot', 'VarP'],
            "set4": ['Max Earliest Time', 'Min Earliest Time', 'MaxP', 'Sum of Max Distance'],
            "set5": ['Max Earliest Time', 'Min Earliest Time', 'MaxP', 'Sum of Max Distance', 'MinM', 'Sum Distance',
                     'Minimum Distance to Depot', 'Average Time Window', 'Average Latest Time', 'Maximum Distance to Depot',
                     'Mean Time Window', 'Percentile 50 Distance', 'SumMaxP', 'Average Distance to Depot', 'Total Number of Nodes'],
            "set6": [ 'MinM','Mean Time Window', 'Percentile 50 Distance', 'SumMaxP',
                      'Average Distance to Depot', 'Total Number of Nodes'],
            "set7": ['Max Earliest Time', 'Mean Time Window', 'Percentile 50 Distance', 'SumMaxP',
                     'Average Distance to Depot', 'Total Number of Nodes'],
            "set8": ['Max Earliest Time', 'Min Earliest Time', 'VarP', 'SumMinP', 'SumMaxP', 'Average Distance to Depot',
                     'Total Number of Nodes'],
            "set9": ['Max Earliest Time', 'Min Earliest Time', 'Sum Distance', 'Total Time Window',
                      'Standard Deviation of Distance to Depot', 'Minimum Distance to Depot', 'Average Time Window',
                      'Average Latest Time', 'Maximum Distance to Depot'],
        
            "set10": ['Max Earliest Time', 'Min Earliest Time'], #top 2 rf features
    
            "set11": ['Max Earliest Time', 'Min Earliest Time', 'MaxP','Sum of Max Distance','MinM' ], #top rf features
            "set12": ['Standard Deviation of Distance to Depot','Max Earliest Time','Minimum Distance to Depot',
                                       'Average Distance to Depot', 'Sum of Distance to Depot'], # top nn features
            "set13": ['Max Earliest Time', 'Sum of Max Distance','Minimum Distance to Depot', 'MaxP',
                                        'Standard Deviation of Distance to Depot'], # top svr features
            "set14": ['Max Earliest Time', 'Standard Deviation of Distance to Depot','Min Earliest Time', 'MaxP','MinM'] # top gbm features
            
        }

    def get_feature_set(self, set_name):
        return self.feature_sets.get(set_name, [])
