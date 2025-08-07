using AuthNuget.Security;
using AuthService.Domain;
using AuthService.Monads;
using AuthService.Queries.Seedwork;
using Microsoft.AspNetCore.Identity;

namespace AuthService.Queries.Jwt;

public sealed class GetJwtForCredentialsHandler : IQueryHandler<GetJwtForCredentials, string>
{
    private readonly UserManager<UserPrincipal> _principalManager;

    public GetJwtForCredentialsHandler(UserManager<UserPrincipal> principalManager)
    {
        _principalManager = principalManager;
    }

    public async Task<Result<string>> Handle(GetJwtForCredentials query, CancellationToken cancellation)
    {
        var user = await _principalManager.FindByNameAsync(query.Username);

        if (user is null)
        {
            return Result.Failure<string>("Invalid Username");
        }

        var passwordCheck = await _principalManager.CheckPasswordAsync(user, query.Password);

        if (!passwordCheck)
        {
            await _principalManager.AccessFailedAsync(user);

            return Result.Failure<string>("Invalid Password");
        }

        var role = await _principalManager.GetRolesAsync(user);

        return Result.Success(JwtFactory.CreateJwtToken(query.Username, role.Single(), RsaKeyStorage.Instance.RsaSecurityKey));
    }
}