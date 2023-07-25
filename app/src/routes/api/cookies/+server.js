export async function POST({ request, cookies }) {
	const json = await request.json();
	Object.entries(json).forEach(([k, v]) => {
		cookies.set(k, v, { path: '/' });
	});
	return new Response();
}
