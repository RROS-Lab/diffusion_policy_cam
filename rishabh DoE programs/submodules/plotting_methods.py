def create_subplots(*datasets, **kwargs):
    subplot_count = len(datasets)
    plt.figure(figsize=(20, 8))
    layouts = kwargs.get('layouts')
    titles = kwargs.get('title')
    sup_title = kwargs.get('sup_title')

    if not titles:
        titles = [f'{i+1}' for i in range(subplot_count)]

    if not layouts:
        layouts = [(1, subplot_count, i+1) for i in range(subplot_count)]

    if not sup_title:
        sup_title = ''

    y_lims = ()
    # Determine the common y-axis limits
    # min_y = np.array(datasets).min() #commented out
    # max_y = np.array(datasets).max() #commented out)
    # y_lims = (min_y, max_y) #commented out

    for i, data in enumerate(datasets):
        plt.suptitle(sup_title)
        plt.subplot(*layouts[i])
        # data = [[_i[0],_i[1]] for _i in data] #commented out
        plt.plot(data)
        plt.legend(['X', 'Y', 'Z']) #commented out

        plt.title(titles[i])
        if y_lims:
            plt.ylim(*y_lims)
        plt.tight_layout()