namespace AuthService.Monads;

public static class BindMonadExtensions
{
    public static Result<TResult> Bind<T, TResult>(
        this Result<T> result,
        Func<T, Result<TResult>> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        return bindFunc(result.Content!);
    }

    public static Result<TResult> Bind<T, TResult>(
        this Result<T> result,
        Func<T, TResult> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        try
        {
            var res = bindFunc(result.Content!);

            return Result.Success(res);
        }
        catch (Exception e)
        {
            return Result.Failure<TResult>(e);
        }
    }

    public static Result Bind(
        this Result result,
        Func<Result> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        return bindFunc();
    }

    public static Result Bind(
        this Result result,
        Action bindAction)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        bindAction();

        return result;
    }

    public static Result<T> Bind<T>(
        this Result result,
        Func<Result<T>> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure<T>(result);

        return bindFunc();
    }

    public static Result Bind<T>(
        this Result<T> result,
        Func<T, Result> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        return bindFunc(result.Content!);
    }

    public static Result Bind<T>(
        this Result<T> result,
        Action<T> bindFunc)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        try
        {
            bindFunc(result.Content!);

            return Result.Success();
        }
        catch (Exception e)
        {
            return Result.Failure(e);
        }
    }

    public static async Task<Result<TResult>> BindAsync<T, TResult>(
        this Result<T> result,
        Func<T, Task<Result<TResult>>> bindFuncAsync)
    {
        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        return await bindFuncAsync(result.Content!);
    }

    public static async Task<Result> BindAsync(
        this Task<Result> resultTask,
        Func<Task<Result>> bindFuncAsync)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure(result);

        return await bindFuncAsync();
    }

    public static async Task<Result> BindAsync<T>(
        this Task<Result<T>> resultTask,
        Func<T,Task<Result>> bindFuncAsync)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure(result);

        return await bindFuncAsync(result.Content!);
    }

    public static async Task<Result> BindAsync(
        this Task<Result> resultTask,
        Action bindFuncAsync)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure(result);

        try
        {
            bindFuncAsync();

            return Result.Success();
        }
        catch (Exception e)
        {
            return Result.Failure(e);
        }
    }

    public static async Task<Result> BindAsync(
        this Task<Result> resultTask,
        Func<Task> bindFuncAsync)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure(result);

        try
        {
            await bindFuncAsync();

            return Result.Success();
        }
        catch (Exception e)
        {
            return Result.Failure(e);
        }
    }

    public static async Task<Result<TResult>> BindAsync<TResult>(
        this Task<Result> resultTask,
        Func<Result<TResult>> bindFuncAsync)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        return bindFuncAsync();
    }

    public static async Task<Result> BindAsync<T>(
        this Result<T> result,
        Func<T, Task<Result>> bindFuncAsync)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        return await bindFuncAsync(result.Content!);
    }

    public static async Task<Result> BindAsync<T>(
        this Result<T> result,
        Func<T, Task> bindFuncAsync)
    {
        if (result.IsFailure()) return Result.FromFailure(result);

        try
        {
            await bindFuncAsync(result.Content!);

            return Result.Success();
        }
        catch (Exception e)
        {
            return Result.Failure(e);
        }
    }


    public static async Task<Result<TResult>> BindAsync<T, TResult>(
        this Task<Result<T>> resultTask,
        Func<T, Task<Result<TResult>>> bindFunc)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        return await bindFunc(result.Content!);
    }

    public static async Task<Result> BindAsync<T>(
        this Task<Result<T>> resultTask,
        Action<T> bindFunc)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure(result);

        try
        {
            bindFunc(result.Content!);

            return Result.Success();
        }
        catch (Exception e)
        {
            return Result.Failure(e);
        }
    }

    public static async Task<Result<TResult>> BindAsync<T, TResult>(
        this Task<Result<T>> resultTask,
        Func<T, Result<TResult>> bindFunc)
    {
        var result = await resultTask;

        if (result.IsFailure()) return Result.FromFailure<TResult>(result);

        return bindFunc(result.Content!);
    }
}