import { fetchPosts } from '$lib/utils.js';
import { json } from '@sveltejs/kit';

export const GET = async () => {
	let posts = await fetchPosts();
	posts = posts.sort((p, q) => new Date(q.metadata.date) - new Date(p.metadata.date));
	return json(posts);
};
