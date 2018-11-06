"""

./demo.py - Method that are using in this file
"""
    
    
        
    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        """

        lines - lyrics

        """
        #create a set of alphabet
        valid_char_set = set(drawing.alphabet)

        #This code validate 2 things
        #1 whether the line is more than 75 char
        #2 whether the charset is invalid

        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

    strokes = self._sample(lines, biases=biases, styles=styles)

    #<-- Go back to the _sample function

    
     def _sample(self, lines, biases=None, styles=None):
         #Generate num_of_samples though line, max_steps, biases
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        #Make zero matricees of num_samples
        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        #if style is pre-define in function

        if styles is not None:
            #takes style number from function and then load those style from npy array
            #where x_p is strokes and c_p is charachter
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')
                
                """
                encodes ascii string to array of ints
                """
                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)
        #if no then generate lines strings to arrays of ints and save it to chars variable
        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)
        #run the nn with sequence and those variables
        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        #normalize that sample and return that sample
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples


    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        #set strokes color and width
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)
        
        #set line height and image dimension
        
        line_height = 60
        view_width = 1000
        view_height = line_height*(len(strokes) + 1)
        
        #create a white background svg file
        
        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))
        #initialize default co-ordinatore
        initial_coord = np.array([0, -(3*line_height / 4)])

        #start drawing!!!!!!
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
            #no line means no drawing :) 
            if not line:
                initial_coord[1] -= line_height
                continue
            # ########

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height
        #save draw
        dwg.save()

    if __name__ == '__main__':
    #create hand object
    hand = Hand()

    print("Starting!")
    # demo number 1 - fixed bias, fixed style
    lines = lyrics.blindly.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]
    """
    Hand()
    Create a neural network object with this parameter
     nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        # And then restore it..
        nn.restore()
    
    """
    """
        hand.write(filename, lines, biases, styles) -->
        sample(lines, biases=biases, styles=styles) which generate strokes from style
        &&
        draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths) -->
    """
    hand.write(
        filename='img/demo1_1.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    print("Lyrics 1")

    lines = lyrics.higlighted.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/demo1_2.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 2")
    lines = lyrics.nextTime.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/demo1_3.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 3")
    lines = lyrics.trYKina.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/demo1_4.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 4")
    lines = lyrics.anotherday.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/demo1_5.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 5")
    # demo number 2 - fixed bias, varying style
    lines = lyrics.people.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/demo2_1.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 6")
    lines = lyrics.living_in_dreams.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/demo2_2.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 7")
    lines = lyrics.why_cant_i.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/demo2_3.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 8")
    lines = lyrics.nevernever.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/demo2_4.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 9")
    lines = lyrics.somedays.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/demo2_5.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 10")
    # demo number 3 - varying bias, fixed style
    lines = lyrics.whatissaid.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/demo3_1.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 11")
    lines = lyrics.intheory.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/demo3_2.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 12")
    lines = lyrics.wandering.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/demo3_3.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 13")
    lines = lyrics.runningaway.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/demo3_4.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 14")
    lines = lyrics.missingyou.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/demo3_5.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
    print("Lyrics 15")
    