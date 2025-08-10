using System.Security.Claims;

namespace AuthService.Extensions;

internal static class ControllerHelpers
{
    public static string GetUsername(this ClaimsPrincipal user)
    {
        return user.Claims.Single(claim => claim.Type.Equals(ClaimTypes.NameIdentifier)).Value;
    }
}