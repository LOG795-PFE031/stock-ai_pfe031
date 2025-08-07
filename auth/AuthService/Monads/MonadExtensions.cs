namespace AuthService.Monads;

public static class MonadExtensions
{
    public static Result<TResult> Failed<TResult>(
        this Result<TResult> result,
        Func<Result<TResult>, Result<TResult>> bindFunc)
    {
        if (result.IsFailure())
        {
            return bindFunc(result);
        }

        return result;
    }

    public static Result<TResult> Failed<TResult>(
        this Result<TResult> result,
        Func<Result<TResult>> bindFunc)
    {
        if (result.IsFailure())
        {
            return bindFunc();
        }

        return result;
    }

    public static Result<TResult> Failed<TResult>(
        this Result<TResult> result,
        Action bindAction)
    {
        if (result.IsFailure())
        {
            bindAction();
        }

        return result;
    }

    public static Result<TResult> Failed<TResult>(
        this Result<TResult> result,
        Func<TResult> bindAction)
    {
        if (result.IsFailure())
        {
            return Result.Success(bindAction());
        }

        return result;
    }

    public static async Task<Result<TResult>> FailedAsync<TResult>(
        this Task<Result<TResult>> resultTask,
        Func<Result<TResult>> bindFunc)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            return bindFunc();
        }

        return result;
    }

    public static async Task<Result> FailedAsync(
        this Task<Result> resultTask,
        Func<Exception, Task<Result>> bindFunc)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            return await bindFunc(result.Exception!);
        }

        return result;
    }

    public static async Task<Result> FailedAsync(
        this Task<Result> resultTask,
        Action<Exception> action)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            action(result.Exception!);
        }

        return result;
    }

    public static async Task<Result> BindForeachAsync<T>(
        this Result<List<T>> result,
        Func<T, Task<Result>> operation)
    {
        if (result.IsFailure())
        {
            return result;
        }

        foreach (var item in result.Content!)
        {
            var innerResult = await operation(item);

            if (innerResult.IsFailure())
            {
                return result;
            }
        }

        return Result.Success();
    }

    public static async Task<Result> BindForeachAsync<T>(
        this Task<Result<List<T>>> resultTask,
        Func<T, Task<Result>> operation)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            return result;
        }

        foreach (var item in result.Content!)
        {
            var innerResult = await operation(item);

            if (innerResult.IsFailure())
            {
                return result;
            }
        }

        return Result.Success();
    }

    public static async Task<Result> BindForeachAsync<T>(
        this Task<Result> resultTask,
        IEnumerable<T> enumerable,
        Func<T, Task<Result>> operation)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            return result;
        }

        foreach (var item in enumerable)
        {
            var innerResult = await operation(item);

            if (innerResult.IsFailure())
            {
                return result;
            }
        }

        return Result.Success();
    }

    public static async Task<Result> BindForeachAsync<T>(
        this Task<Result<IEnumerable<T>>> resultTask,
        Func<T, Task<Result>> operation)
    {
        var result = await resultTask;

        if (result.IsFailure())
        {
            return result;
        }

        foreach (var item in result.Content!)
        {
            var innerResult = await operation(item);

            if (innerResult.IsFailure())
            {
                return result;
            }
        }

        return Result.Success();
    }
}