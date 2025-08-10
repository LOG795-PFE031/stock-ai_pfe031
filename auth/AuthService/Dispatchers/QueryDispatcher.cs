using AuthService.Monads;
using AuthService.Queries.Seedwork;

namespace AuthService.Dispatchers;

public class QueryDispatcher(IServiceProvider serviceProvider, ILogger<QueryDispatcher> logger) : IQueryDispatcher
{
    public async Task<Result<TQueryResult>> DispatchAsync<TQuery, TQueryResult>(TQuery query, CancellationToken cancellation)
        where TQuery : IQuery
    {
        try
        {
            var handler = serviceProvider.GetRequiredService<IQueryHandler<TQuery, TQueryResult>>();

            if (query.GetType().Name is { } queryName && string.IsNullOrWhiteSpace(queryName) is false)
            {
                logger.LogTrace($"Dispatching Query '{queryName}'");
            }

            var queryResult = await handler.Handle(query, cancellation);

            if (queryResult.IsFailure())
            {
                logger.LogError(queryResult.Exception, $"Error while handling query: '{query.GetType().Name}'");
            }

            return queryResult;
        }
        catch (Exception e)
        {
            logger.LogError(e, $"Error while handling query: '{query.GetType().Name}'");

            return Result.Failure<TQueryResult>(e);
        }
    }
}