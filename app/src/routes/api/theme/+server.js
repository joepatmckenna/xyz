import { json } from '@sveltejs/kit';

const KEY = 'darkMode';

export function GET({ cookies }) {
	const darkMode = cookies.get(KEY) || false;
	return json(darkMode);
}

export async function POST({ request, cookies }) {
	const { darkMode } = await request.json();
	cookies.set(KEY, darkMode, { path: '/' });
	return json(darkMode);
}
