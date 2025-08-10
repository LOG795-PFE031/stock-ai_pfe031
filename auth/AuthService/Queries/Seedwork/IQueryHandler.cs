using AuthService.Monads;

namespace AuthService.Queries.Seedwork;

public interface IQueryHandler<in TQuery, TQueryResult> where TQuery : IQuery
{
    Task<Result<TQueryResult>> Handle(TQuery query, CancellationToken cancellation);
}