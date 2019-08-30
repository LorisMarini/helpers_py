from helpers.utils import *


def plot_mono_variate_fit(fit, fig_size=[5, 3], saveas=None, verbose=False, thresholds=None):
    """
    Generates a grid of 2D plots of the statistical model built.
    Parameters
    ----------
    fit :       pd.DataFrame
                A pandas DataFrame that contains the mono-variate model to plot.
    fig_size :  list(int)
                Size of single figure in the multiplot.
    saveas :    [str, None]
                Absolute path where to save the plot.
    verbose :   bool
                Increases output verbosity.
    thresholds: pd.DataFrame
                Contains values used to find outliers.
    Returns
    -------
    fig  :      None, matplotlib.figure.Figure
                Figure containing the results of the analysis.
    """
    check_type(fit, pd.DataFrame)
    check_type(fig_size, list)
    check_type(saveas, [str, type(None)])
    check_type(verbose, bool)
    check_type(thresholds, [pd.DataFrame, type(None)])

    if 'model' not in fit:
        message = f'model not found in fit. Returning None.'
        report_message(message, verbose=verbose, log=True, level='info')
        return None

    # Total number of tiles in the figure
    number_of_plots = len(fit.model.columns)

    # Create subplots
    fig, axarr = plot_grid(number_of_plots, fig_size=fig_size, vs=0.3, hs=0.6, verbose=False)

    # For each kernel, plot kernel on indicated axis
    for i, col in enumerate(fit.model.columns):

        # Extract model kernel
        kernel = fit.loc[:, ('model', col)].iloc[0]
        this_axis = axarr[i]

        if kernel is None:
            # Do nothing
            pass
        else:
            if thresholds is None:
                # Just plot the kernel as is.
                _ = plot_kernel(kernel, axis=this_axis, title=col, thresholds=thresholds)
            else:
                # Plot kernel as well as highlighting the areas indicated by thresholds.
                cdf_1 = thresholds.loc[col, 'cdf_min']
                cdf_2 = thresholds.loc[col, 'cdf_max']

                x_cdf_1 = get_x_from_pcdf(kernel, cdf_1,  verbose=verbose)
                x_cdf_2 = get_x_from_pcdf(kernel, cdf_2,  verbose=verbose)

                _ = plot_kernel(kernel, axis=this_axis, title=col, thresholds=[x_cdf_1, x_cdf_2])

    if saveas is not None:
        fig.savefig(saveas, format='pdf')
        report_message(f'Plot saved as {saveas}', verbose=verbose, log=True, level='info')
    else:
        report_message('Plot not saved.', verbose=verbose, log=True, level='info')

    return fig


def plot_kernel(kernel, axis=None, title=None, on_dataset=False, thresholds=None, **kwargs):
    """
    Plots gaussian_kde (kernel).
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    Parameters
    ----------
    kernel :    scipy.stats.kde.gaussian_kde
                Representation of a kernel-density estimate using Gaussian kernels.
    axis :      None, matplotlib.axes._subplots.AxesSubplot
                matplotlib axis where kernel should be plot.
    title :     str
                Figure title
    on_dataset: bool
                Whether or not we should use the same data in the kernel.dataset to
                plot the probability density function.
    thresholds: None, list(float)
                If not None, is a list of two numbers (thresholds) th1, th2, so that
                any number larger than th2 or smaller then th1 is highlighted.
    Returns
    -------
    """
    check_type(title, [type(None), str])
    check_type(on_dataset, bool)
    check_type(thresholds, [type(None), list])

    # If thresholds is not None, check that it contains floats.
    if isinstance(thresholds, list):
        for t in thresholds:
            check_type(t, float)

    # create axis if not specified.
    if axis is None:
        fig, axis = plt.subplots(**kwargs)
    else:
        axis = axis

    # Remove background color
    axis.patch.set_visible(False)
    sample_size = None

    # Set font size
    matplotlib.rcParams.update({'font.size': 14})

    if kernel is None:
        # Plot a line in zero.
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        axis.plot(x, y, '-')
    else:
        dataset_points = np.sort(np.ravel(kernel.dataset))
        sample_size = len(dataset_points)

        if on_dataset:
            x = dataset_points
        else:
            dataset_min = kernel.dataset.min()
            dataset_max = kernel.dataset.max()
            x = np.linspace(dataset_min, dataset_max, 100)

        # Extract the kernel probability density function
        y = kernel.pdf(x)

        # Plot the estimated density
        axis.plot(x, y, '-', label=f"kernel = '{kernel}'")

        if thresholds is not None:

            th1, th2 = thresholds

            # Fill area under the curve at specified points
            axis.fill_between(x, 0, y, where=x < th1, facecolor='red', alpha=0.5)
            axis.fill_between(x, 0, y, where=x > th2, facecolor='red', alpha=0.5)

        w = np.abs(x.max() - x.min()) * 0.001
        # Plot the original data as well
        axis.bar(dataset_points, np.ones(len(dataset_points)) * 0.2 * max(y), w)

    # Set labels
    axis.set_xlabel('x')
    axis.set_ylabel('p(x) p.d.f.')
    axis.set_xlim(x.min() * 0.9, x.max() * 1.1)
    plt.show()

    if title:
        axis.set_title(title + f', ss = {sample_size}')

    return axis


def plot_multivariate_normal(model, model_input=None, colormap='BuPu', size=[10, 6], saveas=None, verbose=False):
    """
    One liner description
    Parameters
    ----------
    model   :       scipy.stats._multivariate.multivariate_normal_frozen
    model_input :   pd.DataFrame
                    The pandas dataframe with as many column as the multivariate model has, containing
                    the numeric ranges over which we should plot the model probability density function (p.d.f).
    colormap :      str
    size :          list(int)
    saveas :        str
                    Absolute path to a pdf file to disk.
    verbose :        bool
                    Improves output verbosity.
    Returns
    -------
    fig :           matplotlib figure

    Example
    -------
    # stack them so that you get a 2 column matrix
    array = np.vstack((np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))).T
    model_input = pd.DataFrame(array, columns=['x', 'y'])

    # Toy mdoel with null mean, unitary standard deviation and no cross-correlation
    model = multivariate_normal(mean=[np.mean(x),np.mean(y)], cov=[[1,0],[0,1]])

    # Plot model on array values
    figure = plot_multivariate_normal(model, model_input)
    """

    check_type(model, [type(multivariate_normal()), type(None)])
    check_type(model_input, [type(None), pd.DataFrame])
    check_type(colormap, str)
    check_type(size, list)
    check_type(saveas, [type(None), str])
    check_type(verbose, bool)

    if not model:
        message = f"Model None, cannot plot."
        report_message(message, verbose=verbose, log=True, level='info')

        return None

    if not model_input:
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        # stack them so that you get a 2 column matrix
        model_input = pd.DataFrame(np.vstack((x, y)).T, columns=['x', 'y'])

    # Extract values from df
    array = model_input.values
    check_numeric(array, numeric_kinds='uif')

    # Check that dimensions of array and model agree
    if model.dim != len(array.shape):
        raise ValueError(f'Dimensions must agree. model has {model.dim} dimension, '
                         f'while array has {len(array.shape)} dimensions.')

    if model.dim != array.shape[1]:
        raise ValueError('array is expected to have as '
                         'many columns as the dimension in '
                         'the model. Consider transposing it.')

    # Name of what we are plotting:
    variable_names = model_input.columns.values

    # Extract first and second column
    # and create the mesh
    a_range = array[:, 0]
    b_range = array[:, 1]
    a_range_mesh, b_range_mesh = np.meshgrid(a_range, b_range)

    # stack mesh and evaluate model probability density function on dstack
    evaluated_model_range = model.pdf(np.dstack((a_range_mesh, b_range_mesh)))
    report_message('Model p.d.f. calculated on indicated data. ', verbose=verbose, log=True, level='info')

    # open a figure and plot
    fig, ax = plt.subplots(figsize=size)
    contour = ax.contourf(a_range, b_range,
                          evaluated_model_range,
                          100,
                          alpha=1,
                          cmap=plt.get_cmap(colormap))

    plt.xlim([a_range.min(), a_range.max()])
    plt.ylim([b_range.min(), b_range.max()])
    fig.colorbar(contour, shrink=0.9)

    plt.title('normalized fit - probability density function (p.d.f.)')
    plt.xlabel(str(variable_names[0]) + ' (a.u.)')
    plt.ylabel(str(variable_names[1]) + ' (a.u.)')

    if saveas:
        fig.savefig(saveas, format='pdf')
        report_message(f'Plot saved in {saveas}', verbose=verbose, log=True, level='info')
    else:
        report_message('Plot not saved.', verbose=verbose, log=True, level='info')

    return fig, ax


def plot_grid(N, onecol=False, fig_size=[2, 3], hs=0.3, vs=0.6, verbose=False, orient='landscape'):
    """
    Given an integer N, it returns a MxP >= N subplot, (figure and array of axis).
    Parameters
    ----------
    N :             int
                    Total number of subplots
    size :          list(int)
                    Size of single figure
    hs :            float
                    The amount of horizontal separation between subplots
    hv :            float
                    The amount of vertical separation between subplots
    verbose :       bool
                    Increases output verbosity
    Returns
    -------
    fig     :       matplotlib.figure.Figure
    axarr   :       list(matplotlib.axes._subplots.AxesSubplot)

    """

    if onecol:
        n_raws = N
        n_cols = 1
    else:
        # Determine number of rows and columns
        n_raws = np.int(np.ceil(np.sqrt(N)))
        n_cols = np.int(np.ceil(N / n_raws))

        if orient == 'landscape':
            # Prefer a landscape layout
            if n_raws > n_cols:
                n_raws, n_cols = n_cols, n_raws
        if orient == 'portrait':
            if n_cols > n_raws:
                n_raws, n_cols = n_cols, n_raws

    # Determine total figure size
    tot_size = fig_size * np.array([n_cols, n_raws])

    # Create subplot and adjust spacings
    fig, axarr = plt.subplots(n_raws, n_cols, figsize=tot_size)

    plt.subplots_adjust(left=0.125,  # the left side of the subplots of the figure
                        right=0.9,  # the right side of the subplots of the figure
                        bottom=0.1,  # the bottom of the subplots of the figure
                        top=0.9,  # the top of the subplots of the figure
                        wspace=hs,  # the amount of width reserved for blank space between subplots
                        hspace=vs)  # the amount of height reserved for blank space between subplots

    if isinstance(axarr, np.ndarray):
        axarr = np.reshape(axarr, (axarr.size))
    else:
        axarr = np.array([axarr])

    message = f'{axarr.shape}'
    report_message(message, verbose=verbose, log=True, level='info')
    plt.close()

    return fig, axarr


def plot_time_series(series, resample_frequency="60T"):
    """

    Parameters
    ----------
    series
    resample_frequency

    Returns
    -------

    EXAMPLES
    ________

    t_zero = pd.Timestamp(year=2018, month=11, day=27)
    ts_un = time_series_uniform(t_zero=t_zero, max_days=5, size=1000)
    plot_time_series(ts_un)

    ts_ln = time_series_lognormal(t_zero=t_zero, mean=0, sigma=1, size=1000)
    plot_time_series(ts_ln)
    """

    check_type(series, pd.Series)

    # Put time sereis over the index of a dataframe with 1 over its column
    f = series.to_frame()
    f["a"] = 1
    f = f.set_index("timestamp", drop=True)

    # Resample frame
    f_res = f.resample(resample_frequency).sum()

    # Plot
    f_res.plot(color='k', alpha=0.5)
    return None


def hv_layout_save(*, layout, saveas, renderer='bokeh'):
    """
    Saves a holoviews Layout object to file locally.
    Parameters
    ----------
    layout
    saveas
    renderer

    Returns
    -------

    """
    check_type(layout, hv.core.layout.Layout)

    # hv.help(tot_chats_time)
    hv_renderer = hv.renderer(renderer)

    # Using renderer save
    hv_renderer.save(layout, saveas)

    return True


def plot_matrix_from_matrix(xy_values, zoom_out=5, npoints=100):
    """
    One liner description
    Parameters
    ----------
    xy_values : np.ndarray (numeric)
    zoom_out :  numeric
    npoints :   numeric

    Returns
    -------
    plotmatrix : type

    Example
    -------
    # Generate some Nx2 random numbers
    matrix = np.random.randn(8,2)

    # generate the plot matrix associated to these numbers
    plotmatrix = plot_matrix_from_matrix(matrix, zoom_out=3, npoints=20)
    """
    check_type(xy_values, np.ndarray)
    check_numeric(zoom_out, numeric_kinds='uif')
    check_numeric(xy_values, numeric_kinds='uif')
    check_numeric(npoints, numeric_kinds='i')

    # Extract first column
    a = xy_values[:, 0]
    # Extract second column
    b = xy_values[:, 1]

    half_a = zoom_out * (a.max() - a.min()) / 2
    half_b = zoom_out * (b.max() - b.min()) / 2

    a_range = np.linspace(-half_a, half_a, npoints)
    b_range = np.linspace(-half_b, half_a, npoints)

    plotmatrix = np.vstack((a_range, b_range)).T

    return plotmatrix

