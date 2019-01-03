# =============================================================================
# UTILITY FUNCTIONS FILE
# =============================================================================

def update_readme(FLAGS, title, path = '', results = {}, parameters_to_skip=[]):
    """
    Update the readme with the new training parameters.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    FLAGS : tf.flags.FLAGSe
        Tensorflow flags containing the parameters used in the training session.
    
    title : string
        String containing the title of the section
    
    path : string
        String containing the path where the readme file is.
        Default : running script location
    
    results : dict
        Dictionary containing the results to be written in the readme (results not the parameters)
        default : No results to be written
        
    parameters_to_skip : list of strings
        List of all the parameters not to be written
        Default : None
        
    """
    parameters =  FLAGS.flag_values_dict()
    length = 20

    f = open('README.md','a')
    lines = []
    
    # setting the title first
    lines.append(f'\n### {title} \n')

    # setting the parameters
    def _find_desc(par, FLAGS):
        helper = FLAGS.get_help()
        begin = helper.find('--'+par)+len(par)+4
        end = helper[begin:].find('(') + begin
        return helper[begin:end].replace('\n','').replace('   ','')
    
    lines.append("| Parameter | Value | Description |\n| --- | --- | --- |\n")
    for k,v in parameters.items():
        if k in parameters_to_skip:
            continue
        desc = _find_desc(k,FLAGS)
        white_spaces_k = ' ' * (length-len(k))
        white_spaces_v = ' ' * (length-10-len(str(v)))
        white_spaces_d = ' ' * (length-len(desc))
        line = f'| {k}{white_spaces_k} | {v}{white_spaces_v} | {desc}{white_spaces_d} |\n'
        lines.append(line)
        
    # setting the results
    for k,v in results.items():
        white_spaces = ' ' * (length-len(k))
        lines.append(f'* {k}:{white_spaces}{v}\n')

    f.writelines(lines)
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
