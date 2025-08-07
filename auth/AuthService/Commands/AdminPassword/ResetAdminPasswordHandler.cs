using AuthService.Commands.Seedwork;
using AuthService.Domain;
using AuthService.Monads;
using Microsoft.AspNetCore.Identity;

namespace AuthService.Commands.AdminPassword;

public sealed class ResetAdminPasswordHandler : ICommandHandler<ResetUserPassword>
{
    private readonly UserManager<UserPrincipal> _userManager;

    public ResetAdminPasswordHandler(UserManager<UserPrincipal> userManager)
    {
        _userManager = userManager;
    }

    public async Task<Result> Handle(ResetUserPassword command, CancellationToken cancellation)
    {
        var user = await _userManager.FindByNameAsync(command.Username);

        if (user is null) return Result.Failure("Admin user not found");

        await _userManager.RemovePasswordAsync(user);

        var result = await _userManager.AddPasswordAsync(user, command.Password);

        if (!result.Succeeded) return Result.Failure("Password reset failed");

        await _userManager.UpdateAsync(user);

        return Result.Success();
    }
}