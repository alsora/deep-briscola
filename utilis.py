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
    FLAGS : tf.flags.FLAGS
        Tensorflow flags containing the parameters used in the training session.
    title : string
        String containing the title of the section.
    path : string
        String containing the path where the readme file is.
        Default : running script location
    results : dict
        Dictionary containing the results to be written in the readme.
        Default : No results to be written
    parameters_to_skip : list of strings
        List of all the parameters not to be written.
        Default : None
        
    """
    parameters =  FLAGS.flag_values_dict()
    length = 20

    f = open('README.md','a')
    lines = []
    
    # setting the title first
    lines.append(f'\n\n\n### {title} \n')

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

    lines.append("\n\n")
    
    f.writelines(lines)
    f.close()
    
    

class SettingB:
    """
    An object to define the settings of the BriscolaGame environment.
    
    Use the print_settings to see all the settings.
    
    >>> s = SettingB()
    >>> s.print_settings()
    """
    
    def __init__(self):
        self._ordered_hand_by_value = None
        self._state_representation = None
        self._all_past_cards = None
    
    
    def print_settings(self):
        length = 40
        print(f"| Parameter{' '*(length-9)} | Value{' '*(length-5)} |")
        print(f"| {'-'*length} | {'-'*length} |")
        for k,v in vars(self).items():
            white_spaces_k = ' ' * (length-len(k))
            white_spaces_v = ' ' * (length-len(str(v)))
            print(f'| {k}{white_spaces_k} | {v}{white_spaces_v} |')
        
    def check_settings(self):
        for k,v in vars(self).items():
            # TODO : decide if all the settings must be set
            if v is None:
                raise NotImplementedError(f'{k} is not set')
        return True    
    
    @property
    def ordered_hand_by_value(self):
        return self._ordered_hand_by_value
    @ordered_hand_by_value.setter
    def ordered_hand_by_value(self,value):
        '''
        The property to control how the hand is represented.
        
        Available choices
        ----------
        'by_value':
            the hand is in descending order by the value taking into account points and the briscola.
        'by_last_card_used':
            the new card is set in the position of the last card played.
        'just_at_the_end'
            the new card is appended to the player hand that is a list of cards
        
        '''
        self._ordered_hand_by_value = value
        
    @property
    def state_representation(self):
        return self._state_representation
    @state_representation.setter
    def state_representation(self,value):
        '''
        The property to control how the state is represented.
        
        Available choices
        ----------
        'hot_on_deck':
            Hot encoding of the card on the entire deck.
        'hot_on_num_seed':
            Hot encoding of the number of the card and the seed.
        '''
        self._state_representation = value
        
    @property
    def all_past_cards(self):
        return self._all_past_cards
    @all_past_cards.setter
    def all_past_cards(self,value):
        '''
        The property to control how the state is represented.
        
        Available choices
        ----------
        True:
            The history is given.
        False:
            Or not.
        '''
        self._all_past_cards = value
    
    
    
    
    
# =============================================================================
# TEST zone
# =============================================================================
    
s = SettingB()    

s.ordered_hand_by_value = 'by_value'
s.state_representation = 'hot_on_num_seed'
s.all_past_cards = False


s.print_settings()

s.check_settings()








































    
    
