using AuthService.Domain;
using AuthService.Monads;
using AuthService.Queries.Seedwork;
using Microsoft.AspNetCore.Identity;

namespace AuthService.Queries.User;

public sealed class GetUserWalletIdHandler : IQueryHandler<GetUserWalletId, string>
{
    private readonly UserManager<UserPrincipal> _principalManager;

    public GetUserWalletIdHandler(UserManager<UserPrincipal> principalManager)
    {
        _principalManager = principalManager;
    }

    public async Task<Result<string>> Handle(GetUserWalletId query, CancellationToken cancellation)
    {
        var user = await _principalManager.FindByNameAsync(query.Username);

        if (user is null) return Result.Failure<string>("User not found");

        return Result.Success(user.WalletId);
    }
}